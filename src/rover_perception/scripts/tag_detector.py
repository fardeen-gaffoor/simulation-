#!/usr/bin/env python3
"""``ros2 run rover_perception tag_detector.py`` - AprilTags on cam_tag.

The implementation is :class:`rover_perception.nodes.tag_detector_node.TagDetectorNode`.
See docs/tags.md.
"""

from rover_perception.nodes.tag_detector_node import main

if __name__ == "__main__":
    raise SystemExit(main())
