"""Clock alignment between a drone's onboard IMU and the arena tracking system.

The drone does not share the arena's PTP clock, so its log timestamps are offset
from arena time and drift slowly (crystal error, typically tens of ppm). Both
sides observe the same rotation, so we compare angular speed, |omega|, which is
the same in any frame: no camera-to-IMU calibration is needed.

1. Gyro gives |omega| directly; poses give it by differencing orientations.
2. Both are smoothed over the same short baseline (20 ms), resampled to a common rate, normalised and cross-correlated (FFT).
3. The correlation peak, refined to sub-sample precision, gives the offset.
4. Repeating that over windows of the flight and fitting a line gives the drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import quaternion as quat
from .model import ClockModel, ImuStream, PoseStream


@dataclass(frozen=True)
class OffsetEstimate:
    offset_ns: float
    score: float


@dataclass(frozen=True)
class ClockEstimate:
    model: ClockModel
    score: float
    windows_used: int


def gyro_angular_speed(imu: ImuStream) -> tuple[np.ndarray, np.ndarray]:
    return imu.t_ns, np.linalg.norm(imu.gyro, axis=1)


def pose_angular_speed(poses: PoseStream, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    q = quat.normalize(poses.orientation)
    dt_ns = poses.t_ns[stride:] - poses.t_ns[:-stride]
    angle = quat.relative_angle(q[:-stride], q[stride:])
    t_mid = poses.t_ns[:-stride] + dt_ns // 2
    return t_mid, angle / (dt_ns * 1e-9)


def _resample(t_ns: np.ndarray, values: np.ndarray, rate_hz: float) -> np.ndarray:
    rel_s = (t_ns - t_ns[0]) * 1e-9
    n = int(np.floor(rel_s[-1] * rate_hz)) + 1
    grid = np.arange(n) / rate_hz
    return np.interp(grid, rel_s, values)


def _standardise(x: np.ndarray) -> np.ndarray:
    std = x.std()
    if std == 0:
        raise ValueError("signal has no variation: the drone needs to rotate to be aligned")
    return (x - x.mean()) / std


def _moving_average(x: np.ndarray, width: int) -> tuple[np.ndarray, int]:
    """Centred moving average without edge effects; returns (values, samples trimmed from the start)."""
    half = max(0, width // 2)
    if half == 0 or x.size <= 2 * half:
        return x, 0
    return np.convolve(x, np.ones(2 * half + 1) / (2 * half + 1), mode="valid"), half


def estimate_offset(
    imu: ImuStream,
    poses: PoseStream,
    rate_hz: float = 1000.0,
    prior_offset_ns: float | None = None,
    max_deviation_ns: float | None = None,
    min_overlap_s: float = 2.0,
    baseline_s: float = 0.02,
) -> OffsetEstimate:
    """Estimates offset_ns such that t_arena = t_device + offset_ns."""
    if len(imu) < 2 or len(poses) < 3:
        raise ValueError("not enough samples to align")

    pose_period_s = float(np.median(np.diff(poses.t_ns))) * 1e-9
    stride = max(1, min(len(poses) - 2, int(round(baseline_s / pose_period_s))))

    ta, a = gyro_angular_speed(imu)
    tb, b = pose_angular_speed(poses, stride)

    a, trimmed = _moving_average(_resample(ta, a, rate_hz), int(round(stride * pose_period_s * rate_hz)))
    b = _resample(tb, b, rate_hz)
    a = _standardise(a)
    b = _standardise(b)

    la, lb = a.size, b.size
    n_fft = 1 << int(np.ceil(np.log2(la + lb)))
    corr = np.fft.irfft(np.fft.rfft(b, n_fft) * np.conj(np.fft.rfft(a, n_fft)), n_fft)

    lags = np.arange(n_fft)
    lags[lags >= lb] -= n_fft
    overlap = np.minimum(la, lb - lags) - np.maximum(0, -lags)

    base_offset_ns = int(tb[0]) - int(ta[0]) - trimmed * (1e9 / rate_hz)
    offsets_ns = base_offset_ns + lags * (1e9 / rate_hz)

    min_overlap = max(3, int(min(min_overlap_s * rate_hz, 0.5 * min(la, lb))))
    allowed = overlap >= min_overlap
    if prior_offset_ns is not None and max_deviation_ns is not None:
        allowed &= np.abs(offsets_ns - prior_offset_ns) <= max_deviation_ns
    if not allowed.any():
        raise ValueError("IMU and pose streams do not overlap enough in the allowed offset range")

    objective = np.full(n_fft, -np.inf)
    objective[allowed] = corr[allowed] / overlap[allowed]
    peak = int(np.argmax(objective))

    refine = 0.0
    left, right = objective[(peak - 1) % n_fft], objective[(peak + 1) % n_fft]
    if np.isfinite(left) and np.isfinite(right):
        denom = left - 2 * objective[peak] + right
        if denom < 0:
            refine = 0.5 * (left - right) / denom

    k = int(lags[peak])
    n_start, n_end = max(0, -k), min(la, lb - k)
    score = float(np.corrcoef(a[n_start:n_end], b[n_start + k : n_end + k])[0, 1])

    return OffsetEstimate(offset_ns=base_offset_ns + (k + refine) * (1e9 / rate_hz), score=score)


def estimate_clock_model(
    imu: ImuStream,
    poses: PoseStream,
    window_s: float = 10.0,
    rate_hz: float = 1000.0,
    max_deviation_ms: float = 20.0,
    min_score: float = 0.5,
) -> ClockEstimate:
    """Estimates offset and drift of the drone clock against the arena clock."""
    whole = estimate_offset(imu, poses, rate_hz=rate_hz)
    reference_ns = int(imu.t_ns[0])

    centres_s: list[float] = []
    offsets_ns: list[float] = []
    scores: list[float] = []
    window_ns = int(window_s * 1e9)
    for start in range(int(imu.t_ns[0]), int(imu.t_ns[-1]), window_ns):
        window = imu.between(start, start + window_ns)
        if len(window) < 2 or (window.t_ns[-1] - window.t_ns[0]) < 0.8 * window_ns:
            continue
        try:
            est = estimate_offset(
                window,
                poses,
                rate_hz=rate_hz,
                prior_offset_ns=whole.offset_ns,
                max_deviation_ns=max_deviation_ms * 1e6,
                min_overlap_s=0.8 * window_s,
            )
        except ValueError:
            continue
        if est.score < min_score:
            continue
        centre_ns = (int(window.t_ns[0]) + int(window.t_ns[-1])) // 2
        centres_s.append((centre_ns - reference_ns) * 1e-9)
        offsets_ns.append(est.offset_ns)
        scores.append(est.score)

    if len(offsets_ns) < 2:
        return ClockEstimate(ClockModel(whole.offset_ns, 0.0, reference_ns), whole.score, len(offsets_ns))

    slope_ns_per_s, intercept_ns = np.polyfit(centres_s, offsets_ns, 1)
    model = ClockModel(
        offset_ns=float(intercept_ns),
        drift_ppm=float(slope_ns_per_s) * 1e-3,
        reference_ns=reference_ns,
    )
    return ClockEstimate(model, float(np.mean(scores)), len(offsets_ns))
