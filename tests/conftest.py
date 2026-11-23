from pathlib import Path

import numpy as np
import pytest

from arena_data.synthetic import simulate_flight, to_device_clock


def write_euroc_sequence(root: Path, imu, poses) -> Path:
    imu_dir = root / "mav0" / "imu0"
    gt_dir = root / "mav0" / "state_groundtruth_estimate0"
    imu_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    with (imu_dir / "data.csv").open("w") as f:
        f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y,w_RS_S_z,a_RS_S_x [m s^-2],a_RS_S_y,a_RS_S_z\n")
        for t, g, a in zip(imu.t_ns, imu.gyro, imu.accel):
            f.write(f"{t}," + ",".join(f"{v:.9f}" for v in (*g, *a)) + "\n")

    with (gt_dir / "data.csv").open("w") as f:
        f.write("#timestamp,p_RS_R_x [m],p_RS_R_y,p_RS_R_z,q_RS_w [],q_RS_x,q_RS_y,q_RS_z\n")
        for t, p, q in zip(poses.t_ns, poses.position, poses.orientation):
            f.write(f"{t}," + ",".join(f"{v:.9f}" for v in (*p, *q)) + "\n")
    return root


@pytest.fixture(scope="session")
def short_flight():
    return simulate_flight(duration_s=40, seed=11)


@pytest.fixture
def euroc_sequence(tmp_path, short_flight):
    imu, poses = short_flight
    device = to_device_clock(imu, 75_000_000, 20.0)
    return write_euroc_sequence(tmp_path / "seq", device, poses), imu
