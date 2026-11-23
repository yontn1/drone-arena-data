import json

from mcap.reader import make_reader

from arena_data.adapters import EurocImuSource, EurocPoseSource
from arena_data.export import export_session
from arena_data.model import Drone, Match, Pilot, Session
from arena_data.pipeline import assemble_flight


def test_session_round_trip(tmp_path, euroc_sequence):
    root, imu_truth = euroc_sequence
    flight = assemble_flight(
        Drone("whoop-07", frame="65mm"), Pilot("pilot-42", skill_level="novice"),
        EurocImuSource(root), EurocPoseSource(root),
    )
    session = Session("2026-11-23-evening", venue="test", matches=[Match("m1", "capture-the-flag", [flight])])

    manifest_path = export_session(session, tmp_path / "out")

    manifest = json.loads(manifest_path.read_text())
    flight_meta = manifest["matches"][0]["flights"][0]
    assert manifest["session_id"] == "2026-11-23-evening"
    assert flight_meta["drone"]["drone_id"] == "whoop-07"
    assert flight_meta["pilot"]["skill_level"] == "novice"
    assert flight_meta["clock"]["drift_ppm"] is not None

    with (manifest_path.parent / "m1.mcap").open("rb") as f:
        messages = list(make_reader(f).iter_messages(topics=["/whoop-07/imu"]))

    assert len(messages) == flight_meta["imu_samples"]
    first_channel = messages[0][1]
    first = json.loads(messages[0][2].data)
    assert first_channel.metadata["log_time_clock"] == "arena"
    assert first["t_device_ns"] == int(flight.imu.t_ns[0])
    assert abs(messages[0][2].log_time - int(imu_truth.t_ns[0])) < 500_000
