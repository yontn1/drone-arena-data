from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _require_sorted(name: str, t_ns: np.ndarray) -> None:
    if t_ns.ndim != 1:
        raise ValueError(f"{name}: timestamps must be 1-D")
    if t_ns.size > 1 and np.any(np.diff(t_ns) <= 0):
        raise ValueError(f"{name}: timestamps must be strictly increasing")


@dataclass(frozen=True)
class Pilot:
    pilot_id: str
    display_name: str = ""
    skill_level: str = "unknown"


@dataclass(frozen=True)
class Drone:
    drone_id: str
    frame: str = ""
    firmware: str = ""


@dataclass
class ImuStream:
    """Onboard IMU samples on the drone's own clock.

    t_ns: int64 device-clock timestamps in nanoseconds.
    gyro: (N, 3) body-frame angular velocity in rad/s.
    accel: (N, 3) body-frame specific force in m/s^2.
    """

    t_ns: np.ndarray
    gyro: np.ndarray
    accel: np.ndarray

    def __post_init__(self) -> None:
        self.t_ns = np.asarray(self.t_ns, dtype=np.int64)
        self.gyro = np.asarray(self.gyro, dtype=np.float64)
        self.accel = np.asarray(self.accel, dtype=np.float64)
        _require_sorted("ImuStream", self.t_ns)
        n = self.t_ns.size
        if self.gyro.shape != (n, 3) or self.accel.shape != (n, 3):
            raise ValueError("ImuStream: gyro and accel must be (N, 3) matching t_ns")

    def __len__(self) -> int:
        return int(self.t_ns.size)

    def between(self, start_ns: int, end_ns: int) -> ImuStream:
        mask = (self.t_ns >= start_ns) & (self.t_ns < end_ns)
        return ImuStream(self.t_ns[mask], self.gyro[mask], self.accel[mask])


@dataclass
class PoseStream:
    """Drone poses from the arena tracking system, on the arena clock.

    t_ns: int64 arena-clock timestamps in nanoseconds.
    position: (N, 3) metres in the arena frame.
    orientation: (N, 4) unit quaternions (w, x, y, z), body to arena.
    """

    t_ns: np.ndarray
    position: np.ndarray
    orientation: np.ndarray

    def __post_init__(self) -> None:
        self.t_ns = np.asarray(self.t_ns, dtype=np.int64)
        self.position = np.asarray(self.position, dtype=np.float64)
        self.orientation = np.asarray(self.orientation, dtype=np.float64)
        _require_sorted("PoseStream", self.t_ns)
        n = self.t_ns.size
        if self.position.shape != (n, 3) or self.orientation.shape != (n, 4):
            raise ValueError("PoseStream: position must be (N, 3) and orientation (N, 4)")

    def __len__(self) -> int:
        return int(self.t_ns.size)


@dataclass(frozen=True)
class ClockModel:
    """Maps drone device time to arena time.

    t_arena = t_device + offset_ns + drift_ppm * 1e-6 * (t_device - reference_ns)
    """

    offset_ns: float
    drift_ppm: float = 0.0
    reference_ns: int = 0

    def to_arena(self, t_device_ns: np.ndarray) -> np.ndarray:
        t = np.asarray(t_device_ns, dtype=np.int64)
        delta = (t - np.int64(self.reference_ns)).astype(np.float64)
        correction = np.round(self.offset_ns + self.drift_ppm * 1e-6 * delta).astype(np.int64)
        return t + correction

    def to_dict(self) -> dict:
        return {
            "offset_ns": self.offset_ns,
            "drift_ppm": self.drift_ppm,
            "reference_ns": int(self.reference_ns),
        }


@dataclass
class Flight:
    drone: Drone
    pilot: Pilot
    imu: ImuStream
    poses: PoseStream
    clock: ClockModel | None = None
    alignment_score: float | None = None


@dataclass
class Match:
    match_id: str
    game_mode: str = ""
    flights: list[Flight] = field(default_factory=list)


@dataclass
class Session:
    session_id: str
    venue: str = ""
    matches: list[Match] = field(default_factory=list)
