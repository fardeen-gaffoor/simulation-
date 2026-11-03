"""The rclpy node of module 6 (one module, one shim, one executable).

Kept apart from the algorithm modules on purpose: everything under
``rover_excavation/*.py`` imports without ROS so it can be unit-tested in
process, and everything here needs ``rclpy``, the message packages and a running
DDS graph.  ``scripts/excavation.py`` is a three-line shim so the file that gets
installed as an executable is obvious, and the node class stays importable (and
therefore testable) as a normal Python module.

``excavation_node``    serves the ``WorkCycle`` action module 5 calls, runs the
                       mechanism, models the sand, and publishes
                       ``MechanismStatus`` and ``BermStatus``
"""

__all__ = ["excavation_node"]
