# Labelling checklist — real photographs

Read this **before** you label the first real photo. The simulator labels itself
perfectly; the real photographs are the only place a human can introduce a
systematic error, and a systematic error in 200 photos outnumbers 2000 simulated
frames.

The whole point of these photos is that the final model must work on the rover,
so they must be taken the way the rover takes them.

---

## 0. Before you start (once per camera / mounting)

- [ ] **Same camera, same mounting.** Photos must be taken with the camera that
      will fly, at the height and angle it will be mounted at. Check against the
      generated calibration:

      ```
      ros2 run rover_description show_calibration      # or open config/calibration.yaml
      ```

      `extrinsics.translation_xyz[2]` is the camera height above the ground plane.
      If the real height differs by more than 3 cm, **fix `rover.yaml` and
      regenerate** before labelling — `import_real_photos.py --mount-height`
      treats a mismatch as a hard error and will refuse to register the batch.
- [ ] **Same image size and encoding.** Match `image.width` / `image.height` in
      the calibration (`cam_nav` 640x480, `cam_tag` 1280x720, …). A different
      aspect ratio scales the intrinsics and breaks module 4's ground projection.
- [ ] **Note the camera name** you are labelling for (`cam_nav`, `cam_tag`,
      `cam_excavation`, `cam_dump`). One batch = one camera.
- [ ] **Orientation.** The CSI hub may deliver images rotated 180°. Label the
      image **the way it is delivered**, not the way that looks natural. If the
      hub is not yet decided, keep `augment.flip_180.prob` at 0 and revisit.

## 1. What to photograph

- [ ] **Both classes, at the ranges that matter.** Boulders (class 0) and
      craters (class 1). A model trained only on boulders at 2 m will fail on a
      boulder at 6 m.
- [ ] **The real sand**, in the real light: sun low, sun high, overcast, shadow
      edges, wet vs dry if the site can be wet.
- [ ] **Backgrounds that are hard**: sand ripples, tyre tracks, footprints,
      bucket scrape marks, the berm you already built. These must be **labelled
      as empty** (no boxes), otherwise the detector learns to hallucinate.
- [ ] **Partial views**: a boulder half out of frame, a crater at the bottom
      edge. Truncation is allowed (see §3).
- [ ] **Do not photograph the arena markers / AprilTags** as objects. They are
      not in the class list.
- [ ] Aim for **at least 150–300 photos**. Fewer is usually worse than spending
      the same time in the simulator; the sim covers geometry, the real photos
      cover appearance.

## 2. File layout

```
field_photos/batch1/
    0001.jpg
    0001.txt
    0002.jpg
    0002.txt        <- label next to the image (or pass --labels <dir>)
    ...
    labels/         <- optional: 0001.txt, 0002.txt, ...
```

- [ ] One `.txt` per image, same stem, in the same folder or in `--labels`.
- [ ] Labels are **YOLO format**: `class_id cx cy w h`, all four normalised to
      `[0, 1]` from the top-left corner.
- [ ] One line per object, no header, no comments.
- [ ] A background-only photo is an **empty** `.txt` file (0 bytes), not a
      missing file. (Use `--unlabeled-as-negative` only if you really mean it.)

## 3. Boxing rules

- [ ] **Class 0 = boulder.** Box the **visible rock**, not its shadow and not the
      sand it sits in. Include the part that is above the sand surface.
- [ ] **Class 1 = crater.** Box the **rim**, i.e. the whole depression outline —
      the darker ellipse and its raised lip. Do **not** box only the dark centre.
- [ ] **Tight boxes.** The box touches the object. No slack, no padding.
- [ ] **Truncation is fine.** If the object runs off the edge, box the visible
      part and let the box reach the image edge. Do not invent the hidden part.
- [ ] **Occlusion.** If another boulder hides more than roughly half of the
      object, **skip it** — do not box a sliver. Matches the simulator policy
      (`labeling.occluded_policy: drop`).
- [ ] **Size floor.** If a real object is smaller than about 10x10 px, skip it.
      The simulator drops boxes under 6 px; the model cannot learn from smaller.
- [ ] **One box per object.** No duplicates, no nested boxes, no "group" boxes.
- [ ] **Shadows are not objects.** A crater's shadow may be the most visible
      thing about it — box the crater, not the shadow ellipse alone.

## 4. Sanity checks (automated, run them)

```
ros2 run rover_dataset import_real_photos.py \
    --photos field_photos/batch1 --batch batch1 \
    --camera cam_nav --mount-height 0.58

ros2 run rover_dataset split_dataset.py
ros2 run rover_dataset verify_dataset.py --duplicates
```

The importer enforces, per photo:

- [ ] the image is readable and the label file exists,
- [ ] every `class_id` is in `0..nc-1`,
- [ ] every box is inside `[0, 1]` with positive width and height,
- [ ] the stated `--mount-height` matches the calibration within 3 cm.

If **any** of those fails, the importer prints `PROBLEM …`, exits non-zero, and
**does not register the batch**, so a bad batch can never silently reach the
split. The staged files stay under `raw/` for inspection; fix the batch and
re-run.

`verify_dataset.py` then enforces across the whole dataset:

- [ ] every image has a label and every label has an image,
- [ ] every box is a legal YOLO box for the actual image size,
- [ ] `classes.txt` matches `dataset.yaml`,
- [ ] **no scene appears in both train and val**,
- [ ] `--duplicates`: no identical image file in both splits.

## 5. After importing

- [ ] Check the reported **real-photo fraction** from `split_dataset.py`. It must
      reach `sampling.target_real_fraction` (0.60 by default). If it says
      `WARNING: the real-photo target cannot be reached`, add more real photos —
      do not raise `max_repeat` past 8; repeating the same 40 photos eight times
      teaches the model those 40 photos.
- [ ] Look at 20 random training images **after augmentation** (`meta/` sidecars
      list what was applied). If the boxes no longer sit on the objects, stop and
      fix the geometry before training.
- [ ] Re-run the checklist above whenever the camera, the lens or the mounting
      height changes.

## 6. Rules you must not break

- [ ] This dataset is a **training artefact**. Nothing under
      `rover_perception` / `rover_navigation` / `rover_excavation` may read
      `data.yaml`, `labels/` or any ground-truth-derived file at run time. The
      competition forbids giving the autonomy stack obstacle locations.
- [ ] Do not label from video frames that were captured by **driving into** the
      real obstacles; the real arena is graded, and the same rule applies to the
      real site.
