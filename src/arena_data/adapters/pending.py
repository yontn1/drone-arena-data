"""Adapters for the real arena hardware, waiting on hardware decisions.

Each one has the interface the rest of the pipeline uses, so it can be filled in
without touching the core. See "Open questions" in the README.
"""

from __future__ import annotations

from pathlib import Path

from ..model import Drone, ImuStream, PoseStream


class BetaflightBlackboxSource:
    """Reads a Betaflight blackbox log (.bbl/.bfl). Waiting on: flight controller and firmware choice."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def read_imu(self) -> ImuStream:
        raise NotImplementedError(
            "Betaflight blackbox parsing is not built yet: waiting on the flight controller and firmware choice"
        )


class ArenaCameraPoseSource:
    """Reads drone poses from the arena tracking system. Waiting on: camera / tracking system choice."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def read_poses(self) -> PoseStream:
        raise NotImplementedError(
            "Arena camera ingestion is not built yet: waiting on the camera / tracking system choice"
        )


class DroneLogFetcher:
    """Pulls logs off drones after a match. Waiting on: USB, SD card or streamed."""

    def fetch(self, drone: Drone, destination: Path) -> Path:
        raise NotImplementedError(
            "Automatic log download is not built yet: waiting on how logs come off the drone"
        )
