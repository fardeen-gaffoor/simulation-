#!/usr/bin/env python3
"""``ros2 run cat_arena publish_start_pose.py`` - publishes ``/cat_arena/initial_pose``.

Parameters (all settable from a launch file or the command line):

* ``seed``              - run seed; the pose is a pure function of it
* ``config``            - optional ``arena.yaml`` override
* ``frame_id``          - default ``map``
* ``yaw_override_deg``  - force a heading (values beyond +-360 mean "use the config")
* ``also_publish_initialpose`` - mirror onto ``/initialpose`` for RViz
* ``repeat_period_sec`` - republish period (0 publishes once)
"""

from cat_arena.start_pose_node import main

if __name__ == "__main__":
    raise SystemExit(main())
