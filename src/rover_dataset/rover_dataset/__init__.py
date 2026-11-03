"""rover_dataset -- module 3 of the Caterpillar Autonomy Challenge stack.

The perception-dataset factory:

* :mod:`rover_dataset.config`      -- typed ``dataset.yaml`` (the source of truth)
* :mod:`rover_dataset.groundtruth` -- read module 1's ground truth
* :mod:`rover_dataset.calibration_ref` -- read module 2's camera calibration
* :mod:`rover_dataset.projection`  -- ground truth -> YOLO boxes
* :mod:`rover_dataset.augment`     -- per-frame domain randomization
* :mod:`rover_dataset.scene`       -- per-scene world/lighting/mount randomization
* :mod:`rover_dataset.poseplan`    -- capture-pose planning
* :mod:`rover_dataset.capture`     -- gz teleport / pose read-back
* :mod:`rover_dataset.collector`   -- the collection driver + ROS node
* :mod:`rover_dataset.dataset`     -- layout, writing, verification
* :mod:`rover_dataset.split`       -- split by scene + real-photo weighting
* :mod:`rover_dataset.realphotos`  -- import real photographs
* :mod:`rover_dataset.finetune`    -- the fine-tuning recipe
"""

__version__ = "0.3.0"

__all__ = ["__version__"]
