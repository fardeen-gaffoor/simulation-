"""Module 4: camera-only perception for the Caterpillar Autonomy Challenge.

The package is deliberately layered so that every algorithm is importable and
testable without ROS, and only ``scripts/*.py`` touches rclpy:

``config``         typed, validated ``perception.yaml`` (single source of truth)
``geometry``       small rotation/quaternion helpers shared by everything
``calibration_ref``module 2's generated ``calibration.yaml`` (intrinsics/extrinsics)
``markers_ref``    module 1's generated ``markers.yaml`` (wall fiducial poses)
``imagecodec``     ``sensor_msgs/Image`` <-> numpy without cv_bridge/OpenCV
``groundproj``     pixel -> flat-ground -> map-frame point (the module-4 core)
``tagcodes``       the tag36h11 payload table (data)
``tagimages``      AprilTag bitmap rendering (PNG for the sim boards)
``tagboard``       tag board SDF + spawn plan, built from ``markers.yaml``
``tags``           ``tags.yaml``: the runtime tag catalogue (id -> map pose)
``detector``       detector backends (classical now, ONNX/YOLO for the trained model)
``tracking``       multi-frame tracking with confidence decay
``obstacles``      confidence-weighted ``ObstacleArray`` + arena-wall mask
``camera_manager`` phase-based frame rates, bandwidth budget, thermal de-rating
``bandwidth``      sliding-window byte-rate meter (the 4000 kbps rule)
``thermal``        thermal-zone monitoring (sysfs on the Pi, injectable in tests)
``cpu``            fps/CPU measurement without psutil
``localizer``      tag fix -> ``map -> odom`` correction, covariance growth
``deps``           what is missing on this machine and how to install it
"""

__version__ = "0.1.0"
