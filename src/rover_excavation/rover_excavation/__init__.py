"""Module 6: the excavation mechanism and the berm, for the Caterpillar Autonomy Challenge.

The package is layered exactly like modules 4 and 5, so that every algorithm is
importable and testable without ROS and only ``scripts/*.py`` touches rclpy:

``config``     typed, validated ``excavation.yaml`` (single source of truth)
``mechanism``  the ladder/belt/tub kinematics, slew limits and interlocks
``sand``       the tub and the berm height field: where the sand goes
``cycle``      one work cycle (move -> work -> retract), ROS-free and unit-tested
``backend``    how commands reach the actuators: Gazebo joints, serial, or nothing
``nodeutil``   the only module here that imports rclpy

The node is ``excavation`` (started by ``excavation.launch.py`` as
``rover_excavation_mechanism``): it serves the ``WorkCycle`` action that module 5
calls, publishes ``MechanismStatus`` and ``BermStatus``, and draws the berm.

The split with module 5 is the whole point of the module boundary: module 5 owns
*where* and *when*, module 6 owns *how* and *what came of it*.
"""

__version__ = "0.1.0"
