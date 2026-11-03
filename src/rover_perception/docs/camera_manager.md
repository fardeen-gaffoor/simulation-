# The camera manager — phases, the 4000 kbps rule, and thermal throttling

`camera_manager.py` (node `perception_camera_manager`) answers three questions
five times a second:

1. **Which camera needs the full frame rate right now?** The rover is always in
   exactly one phase, and only one camera matters per phase.
2. **Are we inside the comms budget?** The competition measures an average over
   the link; the manager measures the actual bytes arriving and walks a de-rate
   ladder when the average exceeds 4000 kbps.
3. **Is the Pi too hot?** The Raspberry Pi 5 throttles, and a throttled node
   that keeps demanding full rate makes it worse. Thermal state imposes a floor
   on the de-rate ladder.

The manager owns no image data. It subscribes to every camera's topic, counts
bytes and frames (cheaply, without decoding), and publishes a *command* and a
*status*.

---

## Phases

`/perception/mode` (`std_msgs/String`) is driven by module 5. The manager maps a
mode to a role (`config.ACTIVE_CAMERA_ROLE`) and hence to a camera:

| mode | active camera | what the rover is doing |
|---|---|---|
| `idle` | — | parked; everything off |
| `nav` | `cam_nav` | driving |
| `tag` | `cam_tag` | looking for a wall tag to re-anchor |
| `excavate` | `cam_excavation` | digging / berm building |
| `dump` | `cam_dump` | dumping |
| `degraded` | — | every camera at the floor rate, nothing privileged |

The active camera streams at `active_rate_hz` (15 Hz, matching module 2's camera
`update_rate`); every other camera at `idle_rate_hz` (3 Hz — enough to notice
motion, cheap on the link). A camera whose effective rate falls below
`min_rate_hz` (1 Hz) is switched **off** rather than streamed uselessly slowly.

An unknown mode string is ignored with a warning; the last good mode is kept.

## The 4000 kbps rule (`bandwidth.py`)

```yaml
bandwidth:
  limit_kbps: 4000.0
  window_s: 10.0
  hysteresis_kbps: 250.0
  hard_limit_kbps: 6000.0
  derate_factors: [1.0, 0.7, 0.5, 0.35, 0.2, 0.0]
```

`ByteRateMeter` records `(timestamp, n_bytes)` per camera and returns, for each
camera, `bytes * 8 / 1000 / window`. Two details make it honest:

* **The denominator is a wall-clock window, floored at `settle_s` (1 s).** A
  burst of first frames cannot divide by ~0 and trip a spurious de-rate.
* **Samples age out on every tick**, so the number always means "the average over
  the last `window_s` seconds" — which is what the rules actually measure.

The manager walks the de-rate ladder **one rung at a time**:

* measured `> limit` → step **up** (more de-rate), and publish `throttled = true`;
* measured `< limit - hysteresis` → step **down** (less de-rate).

The hysteresis band (`limit - 250 .. limit`) is deliberately dead: without it,
a measured rate hovering at the limit would oscillate every tick. When measured
`> hard_limit_kbps` the manager escalates, but that is a safety net, not the
enforcement path.

Per-camera status (`/perception/camera_status`, `CameraStatusArray`) reports the
*measured* rate and bytes per camera, plus `streaming` (a frame arrived within
`stale_after_s`), `total_kbps`, `over_budget` and `throttled`.

## Thermal (`thermal.py`)

Reads `/sys/class/thermal/thermal_zone*/temp` (auto-detected, or an explicit
`zone_paths` list). Values are millidegrees or degrees; implausible values are
ignored. The **hottest** readable zone is the one used. States:

| state | threshold | effect on the ladder |
|---|---|---|
| `nominal` | `< warn_c` (75 °C) | none |
| `warning` | `>= warn_c` | none (logged) |
| `throttling` | `>= throttle_c` (85 °C) | floor = `throttle_steps` (2 ⇒ factor 0.5) |
| `critical` | `>= critical_c` (95 °C) | floor = last rung ⇒ everything off |

A machine with **no** thermal zones (some containers; not the dev laptop, which
reports `thermal_zone7 TCPU_PCI` at ~100 °C and is therefore always `critical`
here) reports `zone = "none"`, `state = nominal`, and de-rates nothing — a
legitimate, reported condition, not a failure.

The thermal floor can only *raise* the de-rate within a tick; it never silently
un-does the bandwidth ladder. When the Pi cools, the bandwidth logic steps back
down at its own pace.

## The message the driver must honour

`CameraPhase` on `/perception/camera_phase` is the contract with
`rover_camera_driver`:

```
mode, active_camera, active_rate_hz, idle_rate_hz,
cameras[], rate_hz[], enabled[],          # per camera, stable order
bandwidth_ok, measured_kbps, limit_kbps
```

Nothing outside the manager and the driver interprets this. That is deliberate:
whether the CSI hub streams all four cameras at once or round-robins them with a
per-camera dwell **never leaks into perception, navigation or excavation code**.
The driver picks its own strategy to meet the commanded rates.

## In simulation the command is advisory

Gazebo sensors cannot be re-rated at runtime, so in sim the phase is published,
the detector respects the phase for its own publishing rate, and the bandwidth
meter reports what the cameras *really* cost — which is the honest number to test
the budget logic against. The byte accounting is identical in sim and on the
robot, so the meter can be trusted before the real driver exists.

## Health

`health_monitor.py` fuses the manager's `CameraStatusArray`, the detector's
`Float32MultiArray` stats, the localizer's `LocalizationState` and the EKF's
diagnostics into `PerceptionHealth` on `/perception/health` and mirrors it into
`/diagnostics`, at 1 Hz. It reports measured fps, latency, this process's CPU
percent (from `/proc/self/stat`, no `psutil`), the hottest temperature and its
zone, the total bandwidth against the limit, live obstacle count, and the
thermal state — the numbers a judge or an operator wants without `rqt`.

## Tests

```bash
python3 -m pytest src/rover_perception/test/test_bandwidth.py \
                    src/rover_perception/test/test_camera_manager.py \
                    src/rover_perception/test/test_thermal.py \
                    src/rover_perception/test/test_cpu.py -q
```

They pin the settle floor, window ageing, every rung of the ladder, the
hysteresis band, each thermal floor, and the fps/latency/CPU arithmetic against a
synthetic clock and a synthetic `/proc`.
