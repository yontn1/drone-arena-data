from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .model import Drone, ImuStream, PoseStream


class ImuSource(Protocol):
    """Reads one drone's onboard log into an ImuStream on the drone's clock."""

    def read_imu(self) -> ImuStream: ...


class PoseSource(Protocol):
    """Reads one drone's tracked poses into a PoseStream on the arena clock."""

    def read_poses(self) -> PoseStream: ...


class LogFetcher(Protocol):
    """Copies a drone's onboard log to local storage after a match and returns its path."""

    def fetch(self, drone: Drone, destination: Path) -> Path: ...
