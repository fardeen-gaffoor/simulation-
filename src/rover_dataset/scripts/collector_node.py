#!/usr/bin/env python3
"""Data-collection node: teleport the rover, capture, auto-label, write.

    ros2 run rover_dataset collector_node.py --ros-args \
        -p scene_id:=sim_0000 -p ground_truth:=/path/ground_truth.yaml \
        -p calibration:=/path/calibration.yaml -p dataset_root:=dataset

Normally launched by ``collect_dataset.py`` through ``collect.launch.py``; run
this directly only when a simulation is already up.
"""

from rover_dataset.collector import main

if __name__ == "__main__":
    raise SystemExit(main())
