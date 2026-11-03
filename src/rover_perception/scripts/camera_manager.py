#!/usr/bin/env python3
"""``ros2 run rover_perception camera_manager.py`` - phases, rates, thermal, budget.

The implementation is :class:`rover_perception.nodes.camera_manager_node.CameraManagerNode`;
this file exists so the installed executable is obvious and the node class stays
importable by the tests.  See docs/camera_manager.md.
"""

from rover_perception.nodes.camera_manager_node import main

if __name__ == "__main__":
    raise SystemExit(main())
