#!/usr/bin/env python3
"""Generate the rover URDF/SDF, calibration, mass budget and envelope artifacts.

    ros2 run rover_description generate_robot --output-dir /tmp/rover
"""

from rover_description.generate import main

if __name__ == "__main__":
    raise SystemExit(main())
