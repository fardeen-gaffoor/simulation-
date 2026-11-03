# Measured state of one camera, as the manager actually observed it.
# Commanded values live in CameraPhase; these are the facts.

std_msgs/Header header

string name              # camera name, e.g. "cam_nav"
bool streaming           # a frame arrived inside the stale window
bool active              # this is the camera holding the full rate
float32 measured_rate_hz # frames/second actually observed
float32 measured_kbps    # bytes/second actually observed, in kilobits/second
uint32 frames            # frames observed since the manager started
float64 bytes            # payload bytes observed since the manager started
