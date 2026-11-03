"""cat_arena - seeded simulation arena for the Caterpillar Autonomy Challenge.

Public entry points:

* :func:`cat_arena.config.load_config` - read and validate ``arena.yaml``
* :func:`cat_arena.generate.generate` - build a complete seeded arena
* :func:`cat_arena.start_pose.sample_start_pose` - the random spawn pose

The package has no ROS imports at module level, so it can be used from plain
Python, from pytest and from a launch file alike.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
