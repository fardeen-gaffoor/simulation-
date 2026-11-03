# Per-second "is perception healthy?" report: the numbers the judges and the
# thermal/budget logic care about.  Also mirrored into diagnostic_msgs.

std_msgs/Header header

string detector_backend  # "classical", "onnx", "off"
float32 fps              # detector frames per second (measured, not configured)
float32 latency_ms       # mean detector latency over the window
float32 cpu_percent      # this process's CPU use, percent of one core
float32 temperature_c    # hottest monitored thermal zone, degrees Celsius
string thermal_zone      # which zone that was ("none" when unavailable)
float32 bandwidth_kbps   # total camera bandwidth, kilobits/second
float32 bandwidth_limit_kbps
uint32 frames            # detector frames since start
uint32 obstacles         # live tracks in the last ObstacleArray
bool throttled           # camera rates are currently reduced
uint8 thermal_state      # 0 = nominal, 1 = warning, 2 = throttling, 3 = critical
