"""Module 5: navigation and the behaviour sequencer for the Caterpillar Autonomy Challenge.

The package is layered exactly like module 4, so that every algorithm is
importable and testable without ROS and only ``scripts/*.py`` touches rclpy:

``config``     typed, validated ``navigation.yaml`` (single source of truth)
``geometry``   small rotation/quaternion helpers, numpy-only
``costmap``    module 4's obstacle stream -> a remembered, rasterised, inflated map
``planner``    A* on that map, string-pulled into a drivable path
``controller`` pure pursuit with an in-place turn phase (skid-steer)
``footprint``  the exact rotated-rectangle tests for goals and in-place turns
``safety``     the single decision of whether a command may reach the wheels
``mission``    the behaviour sequencer: acquire, dig, haul, dump, park
``nodeutil``   the only module here that imports rclpy

The three nodes are ``navigation`` (map + plan + control), ``velocity_gate``
(the single writer of ``/cmd_vel`` and the safety authority) and ``mission``
(the run's state machine, which drives ``/perception/mode``).
"""

__version__ = "0.1.0"
