# AprilTags — rendering them and reading them back

Module 4 uses AprilTags for one job: **anchoring the map frame**. Tags live on
the marker boards the rules already allow (≤ 60 cm high, ≤ 25×25 cm, ≤ 5 total,
only on the start-zone frame and one wall). There is no AMCL in this stack, so a
tag is the only thing that can ever correct the rover's accumulated drift.

Two halves:

* **solving** — a tag is a catalogue entry the localizer solves against;
* **rendering** — in simulation nothing else puts a texture on module 1's boards,
  so module 4 draws the real bitmaps and spawns them, or the detector would have
  nothing to see.

---

## The catalogue: `markers.yaml` → `tags.yaml`

`cat_arena` emits `markers.yaml`: the physical **boards**, with their pose, face
normal and tag size. Module 4 emits `tags.yaml`: the **tag planes** a detector
actually sees, which are the boards pushed out along their normal by the
standoff.

```
markers.yaml (boards) ──generate_tags.py──► tags.yaml (tags) ─► generate_tags.py ─► bitmaps + SDFs
       ▲ module 1                                                            ▲ module 4
```

The standoff is not a rounding detail. Module 1 draws the dark tag square as a
box of `1.4 × board_thickness` centred on the board, so it protrudes
`0.7 × 0.01 = 7 mm` in front of the board's centre. Module 4 adds a `lift_m`
clearance, so

```
standoff_m = protrusion_m (0.007) + lift_m (0.003) = 0.010 m
```

The **rendered tag plane's** pose is what is recorded in `tags.yaml`, and a test
cross-checks `protrusion_m` against `cat_arena/config/arena.yaml`, so detection
and catalogue agree on the same surface — no systematic bias to remove later.

Each entry in `tags.yaml` records the tag pose in the `map` frame, its size, the
face normal (pointing **into** the arena) and the tag frame's `z` axis. The file
also records the `markers.yaml` path and its sha256, so a run can be audited.

### The tag frame (measured, not assumed)

AprilTag's `T_cam_tag` has the tag's `+z` pointing **into** the tag (away from the
viewer), `+x` to the viewer's right and `+y` down — which is exactly the optical
frame of an observer standing in front of the board. A head-on tag gives
`R == I` from `dt_apriltags`. `tags.tag_frame_rotation(normal, up)` derives that
frame from the board's normal:

```
z = -normal          (into the board)
x = view_forward × up (the viewer's right, view_forward = -normal)
y = z × x
```

This convention is pinned by `test_tagdetect.py::test_head_on_tag_reports_the_viewer_frame`.
`tags.yaml`'s `rpy`/`quaternion_xyzw`/`frame_z` are *this* tag frame; `normal`
and `face_normal` remain the visible face direction **into** the arena.

---

## Rendering the bitmaps

`tagcodes.py` is the tag36h11 code table and the payload→module grid logic, built
from first principles so it can be checked against the published family:

* 587 codes; the code layout is a 1-module quiet zone, an 8-module black square
  with a 1-module black border, and a 6×6 payload inside.
* bit 0 of the payload is the **most significant** module bit.
* `render(texture_size_px=512)` gives `pixels_per_module = 51` and a 510 px
  bitmap; the black square spans `tag_size = 0.18 m`, and the whole bitmap
  (quiet zone included) spans `tag_size * total_modules / 8 = 0.225 m`.
* The bitmap is built by integer module scaling — **no resampling** — so the
  edges are exactly on module boundaries and the detector's decoder is not
  fighting interpolation artefacts.

A test confirms the minimum Hamming distance of the family and that codes 0/1/2
match their known payloads.

`tagimages.py` renders one bitmap or a contact sheet; `preview_tags.py` draws the
sheet so a wrong table or a mirrored layout is visible immediately, and
`test_tagimages.py` makes the real detector read the rendered ids back.

## Spawning a board in simulation

Module 1's boards are **visual-only**, so a running sim has nothing to detect.
`generate_tags.py` writes, per tag:

* `tag_<id>_tag36h11.png` — the bitmap,
* `tag_board_<id>.sdf` — a **static textured `<box>`** visual with
  `<pbr><metal><albedo_map>`, and
* `tag_boards.yaml` — the spawn plan (poses + ready-made `ros_gz_sim create`
  command lines).

> **Measured on gz Harmonic 8.15: a `<plane>` does not texture.** `gz-sim`'s
> plane primitive carries no texture coordinates, so an `<albedo_map>` on a
> `<plane>` renders as one flat colour and the tag can never decode. The board is
> therefore a thin `<box>` (`tagboard._BOARD_THICKNESS_M = 0.002 m`) whose `+z`
> face is textured. The box is centred so the visible face sits *exactly* on the
> catalogue tag plane (`_TAG_FACE_POSITION = (0, 0, thickness/2)`), so folding the
> thickness into the standoff is not needed.

The **model** deliberately carries **no `<pose>`**: the world pose comes only from
`ros_gz_sim create -x -y -z -R -P -Y`, and the visual carries an explicit
face-flip `<pose>` (`tagboard._TAG_FACE_RPY = (π, 0, -π/2)`) because the model
frame is the AprilTag frame whose `+z` points into the board. On a box's `+z`
face a texture maps `u → local -y` and `v → local -x`; the flip compensates.

`perception.launch.py` runs `generate_tags.py` in-process and then spawns one
`ros_gz_sim create` per board after `spawn_delay` seconds. If no `markers.yaml`
exists yet the launch logs exactly how to make one and continues without tag
boards — the detector and the obstacle pipeline still start, so a broken reveal
does not become a mystery.

> If the rendered tag is mirrored it will not decode. The box-face flip is
> `_TAG_FACE_RPY = (π, 0, -π/2)`; the fallback is to swap the in-plane rotation
> (the `-π/2` term) and re-check that a head-on tag still gives `R ≈ I`.

---

## Detecting: `tagdetect.py`

`dt-apriltags` (the real AprilTag 3 / 36h11 detector, working without `cv2`)
wraps into `TagDetector`:

```python
Detector(families, nthreads, quad_decimate, quad_sigma, refine_edges,
         decode_sharpening, debug, searchpath)
detect(img, estimate_tag_pose=False, camera_params=None, tag_size=None)
```

* `tag_size` is a **single value per `detect()` call**, because the underlying
  library takes one; module 4 groups tags by their configured size, calls
  `detect` once per distinct size, and rescales the translation linearly.
* The detector scales the intrinsics from the received `CameraInfo`/image size
  when the frame resolution differs from `calibration.yaml`, and warns once.
* Channels: `decision_margin_min` (15) and `hamming_max` (1) filter junk;
  `max_detections` caps the per-frame list.

`TagDetection` carries the tag's pose in the **camera optical frame** (REP-103:
`+z` forward, `+x` right, `+y` down), plus `decision_margin`, `hamming` and the
four corners in pixels. The localizer is the only consumer, and it is the one
place that knows `tags.yaml`.

A round trip is verified live: a board spawned in a running sim is decoded with
`hamming = 0`, margin ≈ 120, and a translation that matches the spawn pose.

---

## Tools

```bash
ros2 run rover_perception generate_tags.py --out-dir /tmp/opencode/tags   # tags.yaml + bitmaps + SDFs
ros2 run rover_perception generate_tags.py --json
ros2 run rover_perception preview_tags.py                                # draw the bitmaps
ros2 run rover_perception preview_tags.py --ids 0,1,2,3 --out /tmp/opencode/tags.png
```

## Tests

```bash
python3 -m pytest src/rover_perception/test/test_tagcodes.py \
                    src/rover_perception/test/test_tagimages.py \
                    src/rover_perception/test/test_tagboard.py \
                    src/rover_perception/test/test_tags.py \
                    src/rover_perception/test/test_tagdetect.py \
                    src/rover_perception/test/test_markers_ref.py -q
```
