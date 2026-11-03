# The bandwidth meter's verdict: what each camera cost and whether the budget
# was met.  Published on /perception/camera_status.

std_msgs/Header header

CameraStatus[] cameras
float32 total_kbps       # sum of the per-camera measured rates
float32 limit_kbps       # the enforced limit
float32 window_s         # length of the averaging window
bool throttled           # true while a de-rate action is in force
bool over_budget         # true when total_kbps > limit_kbps
