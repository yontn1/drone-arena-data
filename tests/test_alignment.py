import numpy as np
import pytest

from arena_data import ClockModel, ImuStream, estimate_clock_model, estimate_offset
from arena_data.synthetic import simulate_flight, to_device_clock

MAX_ERROR_MS = 0.5


@pytest.fixture(scope="module")
def flight():
    return simulate_flight(duration_s=120, seed=7)


def max_error_ms(model: ClockModel, device: ImuStream, truth: ImuStream) -> float:
    return float(np.abs(model.to_arena(device.t_ns) - truth.t_ns).max()) / 1e6


@pytest.mark.parametrize(
    "offset_ns, drift_ppm",
    [
        (137_400_000, 0.0),
        (-3_210_987_654_321, 0.0),
        (52_300_000, 40.0),
        (-8_000_000, -25.0),
    ],
)
def test_recovers_offset_and_drift(flight, offset_ns, drift_ppm):
    imu, poses = flight
    device = to_device_clock(imu, offset_ns, drift_ppm)

    estimate = estimate_clock_model(device, poses)

    assert estimate.windows_used >= 10
    assert estimate.score > 0.95
    assert estimate.model.drift_ppm == pytest.approx(drift_ppm, abs=2.0)
    assert max_error_ms(estimate.model, device, imu) < MAX_ERROR_MS


def test_offset_only_on_short_flight():
    imu, poses = simulate_flight(duration_s=20, seed=3)
    device = to_device_clock(imu, 250_000_000, 0.0)

    estimate = estimate_offset(device, poses)

    assert estimate.offset_ns == pytest.approx(250_000_000, abs=MAX_ERROR_MS * 1e6)


def test_unrelated_flights_score_low(flight):
    imu, _ = flight
    _, other_poses = simulate_flight(duration_s=120, seed=99)

    estimate = estimate_clock_model(imu, other_poses)

    assert estimate.score < 0.5


def test_drone_that_never_rotates_is_rejected(flight):
    imu, poses = flight
    still = ImuStream(imu.t_ns, np.zeros_like(imu.gyro), imu.accel)

    with pytest.raises(ValueError, match="no variation"):
        estimate_offset(still, poses)


def test_clock_model_round_trip():
    model = ClockModel(offset_ns=1_500_000.0, drift_ppm=30.0, reference_ns=1_000)
    device = np.array([1_000, 1_000 + 10_000_000_000], dtype=np.int64)

    arena = model.to_arena(device)

    assert arena[0] == 1_000 + 1_500_000
    assert arena[1] == 1_000 + 10_000_000_000 + 1_500_000 + 300_000
