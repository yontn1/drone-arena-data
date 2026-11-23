# arena-data

The data pipeline for the AR drone arena. After each match it takes every drone's onboard log and the arena's tracked poses, puts them on one clock, links them to the session, match, drone and pilot, and exports a dataset.

This first version is the hardware-independent core. The pieces that depend on hardware choices are stubs with the same interfaces, so they can be filled in without changing anything else.

## Status

| Part | State |
| --- | --- |
| Data model: session, match, flight, drone, pilot | Done |
| Clock alignment: drone clock to arena clock, offset and drift | Done, tested |
| Dataset export: MCAP per match plus `manifest.json` | Done, tested |
| Reader for public EuRoC MAV sequences, used to develop and test without hardware | Done |
| Reading drone logs (`BetaflightBlackboxSource`) | Waiting on question 1 |
| Getting logs off the drones (`DroneLogFetcher`) | Waiting on question 2 |
| Reading the arena tracking output (`ArenaCameraPoseSource`) | Waiting on question 3 |

### Open questions

1. Which flight controller and firmware (Betaflight?), and so which log format?
2. How do logs come off the drone: USB, SD card, or streamed?
3. Which cameras or tracking system, and what do they output?
4. Do we also want a hardware sync event (for example an LED flash pattern that the cameras see and the drone logs)? It would act as a cross-check on the software alignment below.

## How the clock alignment works

The cameras can share the arena's PTP clock, but the drones can't. A drone's log starts at its own boot time, and its crystal drifts by tens of ppm. At 30 ppm, a 10-minute match drifts by 18 ms, so a single offset isn't enough.

The pipeline aligns the clocks from the flight itself, with no extra hardware:

1. **Same quantity on both sides.** Both the gyro and the tracked orientation see how fast the drone is rotating. Comparing the rotation speed, |ω|, works in any frame, so no camera-to-IMU calibration is needed.
2. **Cross-correlation.** Both signals are smoothed over the same 20 ms, resampled to 1 kHz and cross-correlated with an FFT. The peak, refined to below one sample, gives the clock offset. Any starting offset works, including seconds or hours.
3. **Drift.** The flight is split into 10-second windows, each window is aligned, and a straight line through the results gives the offset and the drift.

The result is a `ClockModel`: `t_arena = t_device + offset + drift_ppm · 1e-6 · (t_device − reference)`.

**Accuracy on simulated flights:**
- 120 s flights with an agile flying style, an offset and ±40 ppm drift: within **0.07 ms** of the true arena time for every sample.
- Very gentle, smooth flying gives a broader correlation peak. Expect up to about 1 ms there.

The method needs the drone to rotate. A drone sitting still can't be aligned, and the pipeline raises an error. A window that doesn't match the poses gets a low score and is left out of the line fit.

It hasn't been run on a real recording yet. The `check-euroc` command below does that on a public dataset.

## Dataset format

```
<out>/<session_id>/manifest.json      session, matches, drones, pilots, clock models, alignment scores
<out>/<session_id>/<match_id>.mcap
    /<drone_id>/imu     {t_device_ns, gyro [rad/s], accel [m/s²]}   logged on the arena clock
    /<drone_id>/pose    {position [m], orientation_wxyz}            logged on the arena clock
```

[MCAP](https://mcap.dev) is the open container format used across robotics tooling, and the files open directly in Foxglove. Messages are JSON for now, so they're easy to read. If the files get too big at 4 kHz, switching to a binary encoding is a change to `export.py` only.

## Running it

Needs Python 3.10+.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

**See the alignment work on a simulated flight with a known clock error:**

```bash
.venv/bin/arena-data demo --offset-ms 137.4 --drift-ppm 30
```

**Run it on a real recording.** Download any sequence of the public EuRoC MAV dataset (ASL format, for example `MH_01_easy`), unzip it, then:

```bash
.venv/bin/arena-data check-euroc path/to/MH_01_easy --offset-ms 137.4 --drift-ppm 30
.venv/bin/arena-data export-euroc path/to/MH_01_easy --out out
```

`check-euroc` shifts the real IMU clock by a known amount and reports how closely the pipeline recovers it.

## Code map

```
src/arena_data/
  model.py        Session, Match, Flight, Drone, Pilot, ImuStream, PoseStream, ClockModel
  sources.py      the interfaces each hardware adapter implements
  adapters/       EuRoC readers (working) and the hardware adapters (stubs)
  alignment.py    clock offset and drift estimation
  pipeline.py     read one drone's log and poses, then align them
  export.py       MCAP + manifest writer
  synthetic.py    simulated flights with known ground truth, for tests and the demo
  cli.py          arena-data command
```
