"""The rclpy nodes of module 4 (one module per node, one shim per executable).

Kept apart from the algorithm modules on purpose: everything under
``rover_perception/*.py`` imports without ROS so it can be unit-tested in
process, and everything here needs ``rclpy``, the message packages and a running
DDS graph.  ``scripts/*.py`` is a three-line shim per node so the file that gets
installed as an executable is obvious, and the node class stays importable (and
therefore testable) as a normal Python module.

The five nodes, and what they own:

``camera_manager_node``   phases, per-camera rates, bandwidth budget, thermal de-rate
``tag_detector_node``     AprilTags on cam_tag -> TagDetectionArray
``tag_localizer_node``    tag fixtures -> map -> odom (+ LocalizationState, TF)
``obstacle_detector_node``boulders/craters on cam_nav -> map-frame ObstacleArray
``health_monitor_node``   fps, CPU, temperature, bandwidth -> PerceptionHealth + diagnostics
"""
