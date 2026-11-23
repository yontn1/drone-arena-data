from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .adapters import EurocImuSource, EurocPoseSource
from .alignment import estimate_clock_model
from .export import export_session
from .model import Drone, Match, Pilot, Session
from .pipeline import assemble_flight
from .synthetic import simulate_flight, to_device_clock


def _report_alignment(imu_truth, device, poses, window_s: float) -> dict:
    estimate = estimate_clock_model(device, poses, window_s=window_s)
    errors_ms = np.abs(estimate.model.to_arena(device.t_ns) - imu_truth.t_ns) / 1e6
    return {
        "estimated_offset_ms": estimate.model.offset_ns / 1e6,
        "estimated_drift_ppm": estimate.model.drift_ppm,
        "score": estimate.score,
        "windows_used": estimate.windows_used,
        "max_error_ms": float(errors_ms.max()),
        "mean_error_ms": float(errors_ms.mean()),
    }


def cmd_demo(args: argparse.Namespace) -> None:
    imu, poses = simulate_flight(duration_s=args.duration, seed=args.seed)
    device = to_device_clock(imu, args.offset_ms * 1e6, args.drift_ppm)
    result = {"true_offset_ms": args.offset_ms, "true_drift_ppm": args.drift_ppm}
    result.update(_report_alignment(imu, device, poses, args.window))
    print(json.dumps(result, indent=2))


def cmd_check_euroc(args: argparse.Namespace) -> None:
    imu = EurocImuSource(args.sequence).read_imu()
    poses = EurocPoseSource(args.sequence, args.pose_source).read_poses()
    device = to_device_clock(imu, args.offset_ms * 1e6, args.drift_ppm)
    result = {"sequence": str(args.sequence), "true_offset_ms": args.offset_ms, "true_drift_ppm": args.drift_ppm}
    result.update(_report_alignment(imu, device, poses, args.window))
    print(json.dumps(result, indent=2))


def cmd_export_euroc(args: argparse.Namespace) -> None:
    flight = assemble_flight(
        Drone(args.drone_id),
        Pilot(args.pilot_id),
        EurocImuSource(args.sequence),
        EurocPoseSource(args.sequence, args.pose_source),
        window_s=args.window,
    )
    session = Session(args.session_id, matches=[Match(args.match_id, game_mode="euroc", flights=[flight])])
    print(export_session(session, args.out))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arena-data")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="align a synthetic flight with a known clock error")
    demo.add_argument("--duration", type=float, default=120.0)
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--offset-ms", type=float, default=137.4)
    demo.add_argument("--drift-ppm", type=float, default=30.0)
    demo.add_argument("--window", type=float, default=10.0)
    demo.set_defaults(func=cmd_demo)

    check = sub.add_parser("check-euroc", help="shift a EuRoC sequence's IMU clock and check it is recovered")
    check.add_argument("sequence", type=Path)
    check.add_argument("--pose-source", default="auto", choices=["auto", "groundtruth", "vicon"])
    check.add_argument("--offset-ms", type=float, default=137.4)
    check.add_argument("--drift-ppm", type=float, default=30.0)
    check.add_argument("--window", type=float, default=10.0)
    check.set_defaults(func=cmd_check_euroc)

    export = sub.add_parser("export-euroc", help="export a EuRoC sequence as an arena dataset")
    export.add_argument("sequence", type=Path)
    export.add_argument("--out", type=Path, default=Path("out"))
    export.add_argument("--session-id", default="euroc-session")
    export.add_argument("--match-id", default="match-1")
    export.add_argument("--drone-id", default="drone-1")
    export.add_argument("--pilot-id", default="pilot-1")
    export.add_argument("--pose-source", default="auto", choices=["auto", "groundtruth", "vicon"])
    export.add_argument("--window", type=float, default=10.0)
    export.set_defaults(func=cmd_export_euroc)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
