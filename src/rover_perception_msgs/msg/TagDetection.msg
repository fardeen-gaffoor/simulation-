# One AprilTag observation on cam_tag, before it is turned into a map fix.
#
# `pose_camera` is the tag's pose in the CAMERA OPTICAL frame (REP-103: +z
# forward, +x right, +y down), which is what the detector measures and what the
# localizer needs.  The tag's physical identity and its map-frame pose come from
# the generated tags.yaml (module 1's markers.yaml handoff).

std_msgs/Header header

int32 id                 # tag id as configured in arena.yaml (0, 1, 2 for the arena)
string family            # e.g. "tag36h11"
float32 size_m           # black-square edge length used for pose estimation, metres
geometry_msgs/PoseWithCovariance pose_camera  # tag pose in the optical frame

float32 decision_margin  # dt-apriltags / apriltag3 decision margin (higher = better)
float64 hamming          # bit errors in the decoded payload (0 = clean)
float32[] corners        # 8 floats: x0 y0 x1 y1 x2 y2 x3 y3, image pixels

bool pose_valid          # detections with a degenerate pose are reported with false
