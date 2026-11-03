#!/usr/bin/env python3
"""``ros2 run rover_perception tag_localizer.py`` - tag fixes -> map -> odom.

The implementation is
:class:`rover_perception.nodes.tag_localizer_node.TagLocalizerNode`.
See docs/localization.md.
"""

from rover_perception.nodes.tag_localizer_node import main

if __name__ == "__main__":
    raise SystemExit(main())
