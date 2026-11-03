"""The rclpy nodes of module 5 (one module per node, one shim per executable).

Kept apart from the algorithm modules on purpose: everything under
``rover_navigation/*.py`` imports without ROS so it can be unit-tested in
process, and everything here needs ``rclpy``, the message packages and a running
DDS graph.  ``scripts/*.py`` is a three-line shim per node so the file that gets
installed as an executable is obvious, and the node class stays importable (and
therefore testable) as a normal Python module.

The three nodes, and what they own:

``navigation_node``    obstacle memory + costmap + planner + controller, and the
                       ``NavigateTo`` action that is the only way anything else
                       asks the rover to move
``velocity_gate_node`` the single writer of ``/cmd_vel``: e-stop, staleness,
                       localisation gating and the reactive brake
``mission_node``       the behaviour sequencer: drives ``/perception/mode``,
                       sends navigation goals and delegates work cycles
"""
