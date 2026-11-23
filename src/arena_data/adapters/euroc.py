"""Readers for the EuRoC MAV dataset layout (ASL format).

A sequence folder looks like:
    <seq>/mav0/imu0/data.csv
    <seq>/mav0/state_groundtruth_estimate0/data.csv
    <seq>/mav0/vicon0/data.csv            (Vicon room sequences only)

Timestamps are nanoseconds. IMU columns: t, wx, wy, wz, ax, ay, az.
Pose columns: t, px, py, pz, qw, qx, qy, qz, ...
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..model import ImuStream, PoseStream


def _mav_root(root: Path) -> Path:
    root = Path(root)
    return root / "mav0" if (root / "mav0").is_dir() else root


def _read_csv(path: Path, columns: int) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    t_ns = np.loadtxt(path, delimiter=",", comments="#", usecols=0, dtype=np.int64, ndmin=1)
    values = np.loadtxt(
        path, delimiter=",", comments="#", usecols=range(1, columns + 1), dtype=np.float64, ndmin=2
    )
    return t_ns, values


class EurocImuSource:
    def __init__(self, root: Path) -> None:
        self.path = _mav_root(root) / "imu0" / "data.csv"

    def read_imu(self) -> ImuStream:
        t_ns, values = _read_csv(self.path, 6)
        return ImuStream(t_ns, values[:, 0:3], values[:, 3:6])


class EurocPoseSource:
    def __init__(self, root: Path, source: str = "auto") -> None:
        mav = _mav_root(root)
        candidates = {
            "groundtruth": mav / "state_groundtruth_estimate0" / "data.csv",
            "vicon": mav / "vicon0" / "data.csv",
        }
        if source == "auto":
            found = [p for p in candidates.values() if p.is_file()]
            self.path = found[0] if found else candidates["groundtruth"]
        elif source in candidates:
            self.path = candidates[source]
        else:
            raise ValueError(f"unknown pose source {source!r}, use auto, groundtruth or vicon")

    def read_poses(self) -> PoseStream:
        t_ns, values = _read_csv(self.path, 7)
        return PoseStream(t_ns, values[:, 0:3], values[:, 3:7])
