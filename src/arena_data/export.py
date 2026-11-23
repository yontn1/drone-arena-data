"""Writes a session as a dataset: one MCAP file per match plus a manifest.json.

MCAP is the open container format used across robotics tooling and can be opened
directly in Foxglove. Every message is logged on the arena clock; IMU messages
also keep the original drone-clock timestamp.

Layout:
    <out>/<session_id>/manifest.json
    <out>/<session_id>/<match_id>.mcap
        /<drone_id>/imu    {t_device_ns, gyro, accel}
        /<drone_id>/pose   {position, orientation_wxyz}
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from mcap.writer import Writer

from .model import Flight, Match, Session

FORMAT_VERSION = 1

IMU_SCHEMA = {
    "type": "object",
    "properties": {
        "t_device_ns": {"type": "integer"},
        "gyro": {"type": "array", "items": {"type": "number"}, "description": "rad/s, body frame"},
        "accel": {"type": "array", "items": {"type": "number"}, "description": "m/s^2, body frame"},
    },
}

POSE_SCHEMA = {
    "type": "object",
    "properties": {
        "position": {"type": "array", "items": {"type": "number"}, "description": "metres, arena frame"},
        "orientation_wxyz": {"type": "array", "items": {"type": "number"}},
    },
}


def imu_topic(drone_id: str) -> str:
    return f"/{drone_id}/imu"


def pose_topic(drone_id: str) -> str:
    return f"/{drone_id}/pose"


def _write_match(path: Path, match: Match) -> list[dict]:
    flights_meta = []
    with path.open("wb") as stream:
        writer = Writer(stream)
        writer.start(profile="", library="arena-data/0.1.0")
        imu_schema = writer.register_schema("arena.Imu", "jsonschema", json.dumps(IMU_SCHEMA).encode())
        pose_schema = writer.register_schema("arena.Pose", "jsonschema", json.dumps(POSE_SCHEMA).encode())

        for flight in match.flights:
            flights_meta.append(_write_flight(writer, flight, imu_schema, pose_schema))

        writer.finish()
    return flights_meta


def _write_flight(writer: Writer, flight: Flight, imu_schema: int, pose_schema: int) -> dict:
    drone_id = flight.drone.drone_id
    clock = "arena" if flight.clock is not None else "device"
    imu_channel = writer.register_channel(
        imu_topic(drone_id), "json", imu_schema, metadata={"log_time_clock": clock}
    )
    pose_channel = writer.register_channel(
        pose_topic(drone_id), "json", pose_schema, metadata={"log_time_clock": "arena"}
    )

    imu = flight.imu
    arena_t = flight.clock.to_arena(imu.t_ns) if flight.clock is not None else imu.t_ns
    for i in range(len(imu)):
        message = {
            "t_device_ns": int(imu.t_ns[i]),
            "gyro": imu.gyro[i].tolist(),
            "accel": imu.accel[i].tolist(),
        }
        t = int(arena_t[i])
        writer.add_message(imu_channel, t, json.dumps(message).encode(), t, sequence=i)

    poses = flight.poses
    for i in range(len(poses)):
        message = {
            "position": poses.position[i].tolist(),
            "orientation_wxyz": poses.orientation[i].tolist(),
        }
        t = int(poses.t_ns[i])
        writer.add_message(pose_channel, t, json.dumps(message).encode(), t, sequence=i)

    return {
        "drone": asdict(flight.drone),
        "pilot": asdict(flight.pilot),
        "topics": {"imu": imu_topic(drone_id), "pose": pose_topic(drone_id)},
        "imu_samples": len(imu),
        "pose_samples": len(poses),
        "clock": flight.clock.to_dict() if flight.clock is not None else None,
        "alignment_score": flight.alignment_score,
    }


def export_session(session: Session, out_dir: Path) -> Path:
    """Writes the session and returns the path of its manifest.json."""
    session_dir = Path(out_dir) / session.session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    matches_meta = []
    for match in session.matches:
        file_name = f"{match.match_id}.mcap"
        flights = _write_match(session_dir / file_name, match)
        matches_meta.append(
            {"match_id": match.match_id, "game_mode": match.game_mode, "file": file_name, "flights": flights}
        )

    manifest = {
        "format_version": FORMAT_VERSION,
        "session_id": session.session_id,
        "venue": session.venue,
        "matches": matches_meta,
    }
    manifest_path = session_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path
