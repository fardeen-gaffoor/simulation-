#!/usr/bin/env python3
"""``ros2 run rover_perception obstacle_detector.py`` - boulders/craters on cam_nav.

The implementation is
:class:`rover_perception.nodes.obstacle_detector_node.ObstacleDetectorNode`.
See docs/detector.md.
"""

from rover_perception.nodes.obstacle_detector_node import main

if __name__ == "__main__":
    raise SystemExit(main())
