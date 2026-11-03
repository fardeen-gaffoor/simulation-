# The mechanism model

`rover_excavation/mechanism.py`. ROS-free, so it is unit-testable anywhere.

It is a *model* of the digger: the physical state module 6 reasons about, and the
state that decides when the rover is allowed to drive. Numbers come from
`excavation.yaml`; the geometry and limits are cross-checked against module 2's
`config/rover.yaml`, which owns the URDF.

## The three actuators

### Ladder pitch — `ladder_pitch_joint`

A revolute joint at `(0.35, 0, 0.15)` in `base_link`, ladder length 0.50 m,
positive pitch tips the tip **down** into the sand (module 2's convention).

| parameter | value | why |
|---|---|---|
| `stowed_pitch_rad` | −0.436 (−25°) | the travel attitude; the envelope check uses it |
| `dig_pitch_rad` | 0.611 (+35°) | the working cut |
| `min_pitch_rad` | −0.524 (−30°) | fully retracted |
| `max_pitch_rad` | 0.960 (+55°) | fully lowered |
| `max_rate_radps` | 0.55 | so deploying takes ≈1.9 s, which is why `MOVING` is a real stage |

Two derived quantities matter:

```
pivot_z_above_sand_m = pivot_z_m + ride_height_m = 0.15 + 0.10 = 0.25 m
min_dig_pitch_rad    = asin(0.25 / 0.50) = 0.524 rad (30.0°)
tip_depth_m(pitch)   = length * sin(pitch) - pivot_z_above_sand_m
```

`pivot_z_m` is where the pivot sits inside `base_link` — the value the URDF uses
— but depth has to be measured from the **sand**, and the drive wheels hold the
body one wheel radius (`ride_height_m` 0.10 m) up. Using `pivot_z_m` alone would
have the belt start biting 0.10 m too early and overstate the cut, and therefore
the dig rate, by roughly 4×. Both terms are copied from `rover_description` and
cross-checked there.

At the dig pitch the tip is **0.037 m** below the sand surface — a shallow
scrape of the tilled layer, which is what a berm is built from.
`min_dig_pitch_rad` is derived rather than configured: below it the tip is simply
in the air and the belt has nothing to bite, so the loader refuses a
`dig_pitch_rad` that is shallower than it. The maximum reachable depth (at +55°)
is 0.160 m, less than the 0.30 m sand layer — the ladder cannot touch the floor
of the arena — and the depth interlock sits just inside that at 0.15 m so the
guard is a guard rather than a number nothing can reach.

### Belt — `belt_joint`

A continuous revolute joint at the ladder tip; roller radius 0.05 m.

| parameter | value |
|---|---|
| `dig_velocity_radps` | +5.0 (sand up the ladder, into the tub) |
| `dump_velocity_radps` | −6.0 (reverse, out of the tub) |
| `max_velocity_radps` | 6.0 (module 2's ceiling) |
| `max_accel_radps2` | 12.0 |

`surface_speed_mps(v) = v * roller_radius` (0.25 m/s while digging) is what the
sand actually sees, and it is what the dig rate is proportional to. A positive
`dump_velocity_radps` in the file is normalised to negative: the sign is not a
tuning knob, reverse is what empties the tub.

### Tub gate

The gate opens over `tub.gate_open_time_s` (0.80 s). A dump does not discharge
until the gate is open **and** the belt is reversing, which is what makes `MOVING`
observable for a dump cycle.

## The interlocks

Two, and both are hard:

* **Depth.** `depth_m > interlocks.max_depth_m` (0.25 m) raises a fault, which
  stows the mechanism. A ladder driven past the sand layer stalls its chain, and
  the loader guarantees `max_depth_m` is both shallower than `arena.sand_depth`
  and reachable at `max_pitch_rad`.
* **Travel.** `travel_ready` is true only in the travel attitude: pitch at or
  above `interlocks.travel_pitch_max_rad`, belt stopped, gate shut. This is the
  fact module 5 needs and cannot compute. A fault clears it too, so a faulted
  rover is never allowed to drive.

A fault is sticky until cleared: the first cause is kept, and `raise_fault` from
anything else does not overwrite it. `hold()` freezes the actuator targets where
they are (used when a cycle is stopped mid-motion); `stow()` aims for the travel
attitude.

## Slew-limited response

The model integrates toward the commanded targets at the limits above rather than
jumping. That is not decoration: it is what makes `MOVING` and `RETRACTING`
observable stages, what makes a work cycle take ~11 s instead of ~7 s, and what
stops a cancelled cycle from teleporting the ladder to the ground and back.

The **node** owns the clock: it calls `Mechanism.step(dt)` once per tick and then
`WorkCycle.step(now)`. A cycle only reads and commands the mechanism; it never
integrates it. That keeps one clock in the process even when the rover sits idle
between cycles.

## The serial protocol

`backend.kind: hardware` speaks to the digger's controller over
`backend.serial.port` at `backend.serial.baud`. The protocol is implemented as
pure `encode_command` / `decode_status` functions in
`rover_excavation/backend.py`, so it is testable without a port.

One command frame per control tick, ASCII, newline terminated:

```
$C,<ladder_pitch_rad>,<belt_velocity_radps>,<tub_gate>*<crc8>\n
```

and one status frame back from the controller:

```
$S,<ladder_pitch_rad>,<belt_velocity_radps>,<tub_gate>,<fault>*<crc8>\n
```

`crc8` is CRC-8/ATM (polynomial `0x07`, initial `0x00`) over everything between
the `$` and the `*`.

The firmware is a thin servo/ESC loop: **it clamps nothing and decides nothing**,
because every limit already lives in `excavation.yaml` and must not be
duplicated in firmware where it can drift. `decode_status` returns `None` for a
malformed frame rather than raising, because a 20 Hz link sees the occasional
torn line and skipping one is not a mechanism failure.

`pyserial` is deliberately **not** a package dependency: install it on the rover
computer only. Without it, `backend.kind: hardware` fails at startup with a
message that says so, rather than silently driving nothing.
