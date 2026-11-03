# rover_dataset — module 3, the perception dataset factory

This package builds the training set for the boulder/crater detector. It has two
jobs:

1. **Simulate and label.** Generate randomized arena+rover scenes, capture frames
   from the rover cameras, and project module 1's ground truth through module 2's
   camera calibration to produce YOLO labels automatically.
2. **Merge the real world.** Import labelled real photographs taken at the real
   camera mounting height, split train/val **by scene**, and make sure the real
   photos carry most of the weight in the final fine-tune.

Nothing here runs on the rover. The dataset is a **training artefact**; the
competition forbids giving the autonomy stack obstacle positions, so nothing
under `rover_perception` / `rover_navigation` / `rover_excavation` may read
`data.yaml`, `labels/` or `scenes.json` at run time.

---

## The pipeline

```
cat_arena.generate ──► ground_truth.yaml ─┐
                                          ├─► projection ─► YOLO boxes ─┐
rover_description.generate ─► calibration ├                             │
                                          │                             ▼
scene randomization ─► arena.yaml/rover.yaml ─► gz world ─► images ─► raw/
                                                                        │
real photos ─► import_real_photos.py ───────────────────────────────────┤
                                                                        ▼
                                                              split_dataset.py
                                                                        │
                                          dataset/images/{train,val} + labels/
                                                                        │
                                                                 finetune.py
```

* **Ground truth** comes from `cat_arena` and is auto-projected; no human labels
  simulated frames. Boulder = bounding box of the sampled, *above-ground* part of
  a partly buried sphere. Crater = bounding box of the projected rim circle.
* **Calibration** comes from `rover_description`, so labels and images can never
  disagree, including when a scene jitters the camera mount.
* **Domain randomization** happens at two levels: per *scene* (sand colour, sun
  elevation/azimuth/tint/intensity, shadows, ambient, background, sand ripples,
  camera-mount jitter) and per *frame* (blur, motion blur, noise, salt & pepper,
  brightness, contrast, gamma, saturation, hue, JPEG, vignette).

## Layout produced

```
dataset/
  raw/images/<scene>__<camera>__p####__f##.jpg     staged, before the split
  raw/labels/<scene>__<camera>__p####__f##.txt
  raw/meta/<scene>__<camera>__p####__f##.json      what was dropped and why
  images/train/...  images/val/...                 symlinks (or --copy)
  labels/train/...  labels/val/...
  meta/train/...    meta/val/...
  preview/grid.png                                 the contact sheet from preview_labels.py
  classes.txt    data.yaml    scenes.json    split.json
  collection.json                                  what the simulator produced
  imports.json                                     what the real photos produced
  _scenes/<scene_id>/{arena.yaml,rover.yaml,scene.json,world/,rover/}
```

`<stem>` is `<scene_id>__<camera>__p<pose>__f<frame>`, so every frame traces back
to the scene, the pose, and the augmentation draw. Repeated real images get an
extra `__r2`, `__r3`, … suffix.

---

## Commands

```bash
source /opt/ros/humble/setup.bash
source ~/cat/install/setup.bash

# 1. collect simulated scenes (writes into dataset/ and builds the split)
ros2 run rover_dataset collect_dataset.py --scenes 8

# a quick smoke run: 2 scenes, 6 poses, one camera
ros2 run rover_dataset collect_dataset.py --scenes 2 --poses 6 --cameras cam_nav

# just write the per-scene configs and stop (inspect before a long run)
ros2 run rover_dataset collect_dataset.py --scenes 4 --dry-run

# 2. import labelled real photographs (one batch = one scene)
#    a batch with any validation problem is rejected: fix it and re-run
ros2 run rover_dataset import_real_photos.py \
    --photos ~/field_photos/batch1 --labels ~/field_photos/batch1/labels \
    --batch batch1 --camera cam_nav --mount-height 0.58

# 3. rebuild the split (after importing real photos)
ros2 run rover_dataset split_dataset.py

# 4. check the dataset
ros2 run rover_dataset verify_dataset.py --duplicates

#    look at it: one PNG with N labelled frames drawn on it
ros2 run rover_dataset preview_labels.py --out-dir dataset --count 16 --columns 4
#    -> write the sheet, then open dataset/preview/grid.png

# 5. print / run the fine-tune recipe
ros2 run rover_dataset finetune.py
ros2 run rover_dataset finetune.py --run        # needs: pip install --user ultralytics

# 6. print the labelling checklist
ros2 run rover_dataset print_checklist.py
```

### Fine-tuning, and why the real photos dominate

`split_dataset.py` prints the real-photo fraction before and after weighting.
Ultralytics has no per-sample loss weights, so extra copies of the real training
images are materialised until real photos are at least
`sampling.target_real_fraction` (default 0.60) of the effective training set:

```
real_repeat = ceil( f * sim * sim_repeat / (real * (1 - f)) )   # capped at max_repeat
```

If that cap cannot reach the target, `split_dataset.py`, `finetune.py` and
`build_training_plan` all say so. Add real photos — do not raise `max_repeat`.

---

## Configuration

Everything lives in `config/dataset.yaml`, validated by `rover_dataset/config.py`.
Changing a value there changes the dataset; no number is hard-coded anywhere
else. `paths.arena_config` / `paths.rover_config` may point at alternative base
configs, and `capture.pose_plan.regions` are the competition zones.

## Testing

```bash
colcon test --packages-select rover_dataset
colcon test-result --verbose
```

Unit tests cover the projection geometry (including a pixel→ground→pixel round
trip), the drop policy, augmentation invariance, scene determinism (and that a
randomized `arena.yaml` still validates in module 1 and a randomized `rover.yaml`
still validates in module 2), the split-by-scene rule, the real-photo weighting
math, the importer's validation, and the whole capture loop against a fake gz
backend.

## Assumptions and open items

* The CSI hub is not chosen yet, so images are **not** assumed rotated:
  `augment.flip_180.prob` and `horizontal_flip.prob` are 0. Set them once the hub
  is known.
* `capture.cameras` defaults to `cam_nav` + `cam_tag`; `cam_excavation` and
  `cam_dump` are available in `rover.yaml` when the excavation classifier needs
  its own data.
* Craters are assumed to be circular and at the mean sand level (`labeling.ground_z`
  = 0); boulders are spheres given by the module 1 ground truth.
* Occlusion only considers boulders blocking boulders/craters. Craters never
  occlude, which is conservative: a base hidden by a crater rim stays labelled.
* `real_repeat` uses symlinks by default; pass `--copy` to `split_dataset.py` to
  produce a self-contained dataset for copying to the training machine.
* `ultralytics` is not a ROS dependency and is not installed by this package.
