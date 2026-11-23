"""Synthetic flights with known ground truth, for testing alignment end to end."""

from __future__ import annotations

import numpy as np

from . import quaternion as quat
from .model import ImuStream, PoseStream

GRAVITY = np.array([0.0, 0.0, -9.81])


def _sinusoids(rng: np.random.Generator, count: int, f_range: tuple, a_range: tuple) -> list:
    return [
        (rng.uniform(*f_range), rng.uniform(*a_range), rng.uniform(0, 2 * np.pi))
        for _ in range(count)
    ]


def _eval(terms: list, t: np.ndarray, derivative: int = 0) -> np.ndarray:
    out = np.zeros_like(t)
    for f, a, phase in terms:
        w = 2 * np.pi * f
        if derivative == 0:
            out += a * np.sin(w * t + phase)
        elif derivative == 2:
            out += -a * w * w * np.sin(w * t + phase)
    return out


def simulate_flight(
    duration_s: float = 60.0,
    imu_rate_hz: float = 1000.0,
    pose_rate_hz: float = 200.0,
    start_ns: int = 1_700_000_000_000_000_000,
    seed: int = 0,
    gyro_noise: float = 0.02,
    accel_noise: float = 0.05,
    orientation_noise_rad: float = 0.001,
    position_noise_m: float = 0.0005,
) -> tuple[ImuStream, PoseStream]:
    """Returns (imu, poses) for the same flight, both on the arena clock."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * imu_rate_hz)
    dt = 1.0 / imu_rate_hz
    t = np.arange(n) * dt

    rate_terms = [_sinusoids(rng, 4, (0.1, 6.0), (0.5, 3.0)) for _ in range(3)]
    pos_terms = [_sinusoids(rng, 3, (0.05, 0.5), (0.2, 1.5)) for _ in range(3)]

    omega = np.stack([_eval(terms, t) for terms in rate_terms], axis=1)
    omega_mid = np.stack([_eval(terms, t + dt / 2) for terms in rate_terms], axis=1)
    steps = quat.from_rotvec(omega_mid * dt)

    q = np.empty((n, 4))
    q[0] = [1.0, 0.0, 0.0, 0.0]
    for k in range(n - 1):
        q[k + 1] = quat.multiply(q[k], steps[k])
        q[k + 1] /= np.linalg.norm(q[k + 1])

    position = np.stack([_eval(terms, t) for terms in pos_terms], axis=1) + np.array([0, 0, 2.0])
    accel_world = np.stack([_eval(terms, t, derivative=2) for terms in pos_terms], axis=1)
    rotation = quat.to_matrix(q)
    specific_force = np.einsum("nji,nj->ni", rotation, accel_world - GRAVITY)

    t_ns = start_ns + np.round(t * 1e9).astype(np.int64)
    imu = ImuStream(
        t_ns,
        omega + rng.normal(0, gyro_noise, omega.shape),
        specific_force + rng.normal(0, accel_noise, specific_force.shape),
    )

    step = int(round(imu_rate_hz / pose_rate_hz))
    idx = np.arange(0, n, step)
    noisy_q = quat.normalize(
        quat.multiply(q[idx], quat.from_rotvec(rng.normal(0, orientation_noise_rad, (idx.size, 3))))
    )
    poses = PoseStream(
        t_ns[idx],
        position[idx] + rng.normal(0, position_noise_m, (idx.size, 3)),
        noisy_q,
    )
    return imu, poses


def to_device_clock(imu: ImuStream, offset_ns: float, drift_ppm: float) -> ImuStream:
    """Re-stamps arena-clock IMU samples as a drone clock that is offset and drifting."""
    t_arena = imu.t_ns
    elapsed = (t_arena - t_arena[0]).astype(np.float64)
    t_device = t_arena - np.round(offset_ns + drift_ppm * 1e-6 * elapsed).astype(np.int64)
    return ImuStream(t_device, imu.gyro, imu.accel)
