# The camera-manager -> camera-driver contract.
#
# rover_camera_driver (the real-hardware package, not yet built) MUST honour this
# message: the camera named in `active_camera` streams at `active_rate_hz`, every
# other managed camera streams at `idle_rate_hz`, and a camera whose entry in
# `rate_hz` is 0.0 is switched off.  Nothing outside rover_perception and
# rover_camera_driver interprets this message, which is what keeps the CSI hub
# model private to the driver.
#
# In simulation the Gazebo sensors cannot be re-rated at runtime, so the manager
# still publishes the phase (the detector respects it) and reports the measured
# bandwidth, but commands are advisory.  See docs/camera_manager.md.

std_msgs/Header header

string mode             # perception phase that asked for this: idle|nav|tag|excavate|dump|degraded
string active_camera    # camera that gets the full rate ("" = every camera idle)

float32 active_rate_hz  # commanded rate of the active camera
float32 idle_rate_hz    # commanded rate of every other streaming camera

string[] cameras        # every managed camera, in a stable order
float32[] rate_hz       # commanded per-camera rate, same order as `cameras`
bool[] enabled          # per-camera on/off, same order as `cameras`

bool bandwidth_ok       # false while the measured average exceeds the competition limit
float32 measured_kbps   # measured average over the manager's window
float32 limit_kbps      # the limit being enforced (competition: 4000 kbps average)
