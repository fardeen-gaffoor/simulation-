# The obstacle detector — boulders and craters, camera-only

Module 4's obstacle pipeline is `obstacle_detector.py`:

```
/cam_nav ─► backend ─► Detection ─► GroundObservation ─► Tracker ─► wall mask ─► ObstacleArray
             ▲                          ▲
        classical | onnx        calibration.yaml + rover pose
```

Everything after the backend is shared. That is the point: module 5 can be
developed against the `classical` backend today and switched to the fine-tuned
model by editing one line of `perception.yaml`, with no change in the obstacle
map's shape or meaning.

---

## Why there is no depth

The competition allows a monocular camera and nothing that measures distance
directly (no depth sensor, no ultrasonic, no lidar). A detection is therefore a
bounding box in one RGB frame, and the only way to place it on the map is the
**flat-ground assumption**: the bottom of the box touches the sand plane
`z = ground_z`, so the pixel defines a ray through the pinhole camera, and that
ray meets the sand plane at a unique point.

This is exactly the convention module 3 used to draw the training labels, so a
model trained on module 3's boxes is consumed with the same ground rule and the
two cannot disagree by a systematic offset.

| Step | Code |
|---|---|
| contact pixel = bbox bottom-centre | `groundproj.contact_pixel` |
| ray through that pixel (calibration + pose) | `unproject_pixel` |
| ray ∩ `z = ground_z` | `unproject_pixel` |
| radius from box width and optical depth | `observe_detection` |
| boulder exposed height from the box top | `height_for_range` |
| reject with a reason, or return an observation | `observe_detection` |

The forward projection (`project_points`) and the inverse (`unproject_pixel`)
are tested against each other on the real calibration, so a sign error cannot
hide in one of them.

### What is deliberately not measured

* **Crater depth is `0.0`.** A single camera looking at uniformly sandy ground
  cannot measure how deep a pit is. Module 4 measures *where* the crater is and
  *how wide* it is, and marks it lethal, so module 5 routes around it. Publishing
  a fake depth would be worse than publishing none.
* **Boulder height is an estimate**, capped at `2 * radius`, because there is no
  way to see behind the object. It is reported for context, not as a guarantee.

### Rejections

Every detection that cannot be placed is dropped with a machine-readable reason,
counted in the stats topic, and never silently ignored:

`contact_pixel_outside_image`, `above_horizon`, `too_near`, `too_far`,
`too_small`, `too_large`, `outside_arena`.

`too_near`/`too_far` are `ground.min_range_m`/`max_range_m`: the flat-ground
model is meaningless below the near limit, and beyond the far limit the
range error grows as `range²`.

---

## The `classical` backend (`backends/classical.py`)

It runs today with numpy only — no model file, no `onnxruntime`, no `cv2` — and
it exists so the whole pipeline is exercisable before the model is trained.

1. **Region of interest.** Only rows below the horizon (`horizon_row`, from the
   camera's extrinsics) are considered; above the horizon there is no sand.
2. **Sand reference.** The sand colour is the median luminance/chroma of a band
   of rows near the bottom of the image, and its robustness (MAD) gives an
   adaptive threshold. A rock casting a shadow is still an anomaly against the
   sand; a grass-free, uniformly-lit beach is not.
3. **Anomaly mask.** A pixel is anomalous when it is darker than
   `reference - delta_luma`, *or* off-colour by more than `delta_chroma`, *or*
   more than `mad_k` robust deviations away.
4. **Connected components.** 8-connected, run-length + union-find, then shape
   gates: `min_area_px`, `min_fill` (area / bbox), `max_aspect`,
   `max_bbox_fraction`.
5. **Classify by shading.** For each blob, split at the vertical midline and
   compare the mean grey of the top half with the bottom half:

   | `grey_top - grey_bottom` | meaning | class |
   |---|---|---|
   | `> +shadow_delta` | lit from above ⇒ convex | boulder |
   | `< -shadow_delta` | shaded on top ⇒ concave | crater |
   | otherwise | ambiguous | `ambiguous_kind` (default **crater**) |

   Ambiguity resolves to a crater on purpose: the cost of treating sand as a hole
   is one cautious detour; the cost of driving into a hole is a failed run. A
   blob touching the region edge is also marked `conservative`.

A test renders a synthetic sand field with a lit top / dark bottom patch and a
dark top / lit bottom patch and asserts both are classified correctly.

### Measured behaviour in the seed-1 arena (and its limits)

The bootstrap backend was checked live against `worlds/seed_1/ground_truth.yaml`
by teleporting the rover to a known pose and running the real pipeline on
captured `cam_nav` frames. The honest summary:

* **Close range works well.** From `(3.6, 0.6, yaw 1.35)` all four boulders that
  were in frame were found, every one classified as a boulder, 0.14–0.23 m from
  the seeded centre. The 0.15 m systematic offset is the documented contact rule:
  the contact pixel is the *near* edge of the footprint, so the estimate is pulled
  toward the camera by about a radius.
* **Long range degrades.** From `(7.9, 1.2, yaw 2.36)` only 2 of 5 boulders were
  recovered at the same accuracy. Two effects cause it, both visible in the mask:
  1. **Distance shading.** The sand median luminance runs 134→182 across the
     ground region (the single global `reference_grey` cannot model that), so in
     the far field a boulder and the sand around it have nearly the same
     luminance and only the *chroma* term separates them.
  2. **The near-wall desaturation band.** The far sand loses saturation until its
     chroma matches the grey boulders, forming one wide horizontal anomaly that
     swallows the boulders embedded in it.
* **Known false positives** are the arena's painted zone lines and the boulders'
  own cast shadows — both are sand-plane luminance/chroma anomalies that no
  unsupervised segmenter can tell from a crater. The painted zone lines are a
  simulation-only visual (the real arena has no tape), and shadow-versus-hole is
  exactly the ambiguity a *learned* classifier resolves; this is why the model is
  the production backend and the segmenter is the bootstrap.
* Craters were not recovered in the sim from any tested vantage: the seeded
  craters are shallow relative to their 0.16–0.24 m radius and do not project the
  strong, symmetric top-dark/bottom-lit signature the classifier looks for.

None of this changes the wire contract: the pipeline, the projection maths, the
rejections, the tracker and the published message are backend-independent, and
every failure above is visible in `/perception/detector_stats` (as a rejection
reason) or in the per-obstacle `conservative` flag — never silent.

---

## The `onnx` backend (`backends/onnx_yolo.py`)

The YOLOv8n model fine-tuned in module 3, exported to ONNX and run on the Pi's
CPU via `onnxruntime` (a Hailo path is a later option, not required). The class
order is fixed by module 3's dataset: **0 = boulder, 1 = crater**.

Everything around the session is hand-written numpy, because it has to be
portable to the Pi and `cv2` cannot be used on this laptop:

| Function | Job |
|---|---|
| `letterbox` | scale-preserving resize onto a square canvas, pad colour exact |
| `resize_bilinear` | the resize itself, no OpenCV |
| `decode_predictions` | `(1, 4+nc, N)` head → xyxy boxes, best class + score |
| `nms` | per-class greedy non-maximum suppression |
| `OnnxYoloDetector` | session + the un-letterboxing back to image pixels |

The class-agnostic boxes are then turned into the same `Detection` objects the
classical backend produces, so tracking and projection never learn which backend
ran. The whole decode path is unit-tested with a **stub session**, so it is
correct before any model exists:

```bash
python3 -m pytest src/rover_perception/test/test_onnx_backend.py -q
```

`perception_info.py` prints whether `onnxruntime` is importable and
`generate_tags`-style tools report the configured model path. With
`detector.onnx.model: ""` the backend refuses to start and tells you what to set.

---

## Tracking (`tracking.py`)

A single frame is noisy: a boulder can be seen in one frame and hidden behind
sand spray in the next. The tracker keeps **one track per physical obstacle**:

* detections are associated to tracks by ground distance
  (`associate_radius_m`), ignoring class;
* a track is published once it has `min_hits` observations and kept for
  `max_misses` missed frames;
* confidence rises with each sighting (`hit_gain`) and decays with a **wall-clock
  half-life** (`decay_half_life_s`), so the decay does not depend on how often
  the node happens to run;
* **kind only ever escalates**: a track seen as a boulder and then as a crater
  becomes a crater (never the reverse) and is flagged `conservative`;
* position is smoothed with `position_gain`;
* ids are class-neutral (`obstacle_<n>`) so reclassifying a track never changes
  its identity, and module 5 can keep its own plans keyed on the id.

`SIGMA_GROWTH_PER_MISS` inflates the position covariance of a track that is not
being re-sighted, so an obstacle last seen three frames ago is published with a
larger uncertainty.

---

## The arena-wall mask (`obstacles.py`)

The competition forbids using the walls as sensors, and the walls are known
geometry, so anything that lands outside the arena or inside
`arena.wall_margin_m` of a wall is not an obstacle: it is a wall, a wall shadow,
or a reflection. `wall_mask_reason` returns one of

`outside_arena_x`, `outside_arena_y`, `inside_wall_margin`

and `mask_tracks` drops those tracks, counting them in `ObstacleArray.dropped_wall`.

## The published obstacle (`ObstacleArray` / `Obstacle`)

`header.frame_id` is always `map`. Every obstacle carries its **evidence**:
`observations`, `confidence` (tracker), `score` (best raw detector score),
`age_s`, `last_seen_age_s`, `source` (backend name) and `conservative`. The
`pose` is the footprint centre on the sand plane with the perception covariance
populated from the range model, so a consumer can weight distant detections
less. `kind` uses the same numbers module 5 and module 3 use
(`KIND_BOULDER = 1`, `KIND_CRATER = 2`).

A `sensor_msgs/PointCloud2` (fields `x y z intensity`) is published on
`/perception/obstacles_cloud` for a nav2 costmap layer; the `ObstacleArray`
remains the authoritative interface.

---

## Configuration

All in `config/perception.yaml`:

* `detector.backend`, `detector.camera`, `detector.publish_hz`;
* `detector.ground.*` — the projection policy and the range/radius gates;
* `detector.classical.*` — the segmentation and classification gates;
* `detector.onnx.*` — model path, providers, letterbox, thresholds;
* `tracking.*` — association, hits/misses, confidence, decay.

## Live check (manual, against the running arena)

```bash
# terminal 1: world + rover + perception
ros2 launch cat_arena arena.launch.py robot_plugins:=true seed:=1
ros2 launch rover_description sim.launch.py
ros2 launch rover_perception perception.launch.py

# terminal 2
ros2 topic echo /perception/obstacles --once
ros2 topic hz   /perception/obstacles
ros2 topic echo /perception/detector_stats --once    # fps, latency, cpu, tracks
```

Move the rover with `/cmd_vel` and confirm each seeded obstacle appears once,
in the right place, with a stable id — and that the arena walls never appear.

> **Ground truth is for comparison only.** `worlds/seed_1/ground_truth.yaml`
> carries the seeded obstacle list. Read it to check the detector; never read it
> from perception. `perception.yaml` duplicates only the arena *geometry* and a
> test cross-checks that copy against `cat_arena/config/arena.yaml`.
