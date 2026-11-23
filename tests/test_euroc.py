import numpy as np
import pytest

from arena_data.adapters import EurocImuSource, EurocPoseSource
from arena_data.model import Drone, Pilot
from arena_data.pipeline import assemble_flight


def test_reads_imu_and_poses(euroc_sequence, short_flight):
    root, _ = euroc_sequence
    _, poses_truth = short_flight

    imu = EurocImuSource(root).read_imu()
    poses = EurocPoseSource(root).read_poses()

    assert len(imu) == 40_000
    assert imu.t_ns.dtype == np.int64
    assert len(poses) == len(poses_truth)
    np.testing.assert_array_equal(poses.t_ns, poses_truth.t_ns)
    np.testing.assert_allclose(poses.orientation, poses_truth.orientation, atol=1e-8)


def test_assembled_flight_is_aligned_to_arena_clock(euroc_sequence):
    root, imu_truth = euroc_sequence

    flight = assemble_flight(Drone("d1"), Pilot("p1"), EurocImuSource(root), EurocPoseSource(root))

    error_ms = np.abs(flight.clock.to_arena(flight.imu.t_ns) - imu_truth.t_ns).max() / 1e6
    assert error_ms < 0.5
    assert flight.alignment_score > 0.95


def test_missing_sequence_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        EurocImuSource(tmp_path).read_imu()


def test_unknown_pose_source_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown pose source"):
        EurocPoseSource(tmp_path, source="lidar")
