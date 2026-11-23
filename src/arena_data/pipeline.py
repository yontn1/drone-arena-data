from __future__ import annotations

from .alignment import estimate_clock_model
from .model import Drone, Flight, Pilot
from .sources import ImuSource, PoseSource


def assemble_flight(
    drone: Drone,
    pilot: Pilot,
    imu_source: ImuSource,
    pose_source: PoseSource,
    align: bool = True,
    window_s: float = 10.0,
) -> Flight:
    """Reads one drone's log and tracked poses for a match and aligns their clocks."""
    imu = imu_source.read_imu()
    poses = pose_source.read_poses()
    flight = Flight(drone=drone, pilot=pilot, imu=imu, poses=poses)
    if align:
        estimate = estimate_clock_model(imu, poses, window_s=window_s)
        flight.clock = estimate.model
        flight.alignment_score = estimate.score
    return flight
