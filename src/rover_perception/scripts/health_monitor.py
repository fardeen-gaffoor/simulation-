#!/usr/bin/env python3
"""``ros2 run rover_perception health_monitor.py`` - fps, CPU, temperature, budget.

The implementation is
:class:`rover_perception.nodes.health_monitor_node.HealthMonitorNode`.
See docs/localization.md and docs/camera_manager.md for the numbers it reports.
"""

from rover_perception.nodes.health_monitor_node import main

if __name__ == "__main__":
    raise SystemExit(main())
