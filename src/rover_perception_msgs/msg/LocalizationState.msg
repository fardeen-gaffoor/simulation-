# The state of the map -> odom correction, published by tag_localizer.
#
# There is no AMCL in this stack: the arena frame is anchored by AprilTags on
# the walls, and this message reports how strongly it is currently anchored, so
# module 5 can decide to slow down (or stop and re-look) while the rover is
# dead-reckoning.  `sigma_*` grow with odometry alone and collapse on a fix.

std_msgs/Header header

bool localized           # true once at least one fix has been accepted
uint32 fixes_total       # accepted fixes since start
uint32 fix_age_frames    # frames since the last accepted fix
float32 time_since_fix_s # seconds since the last accepted fix

float32 sigma_xy_m       # current 1-sigma position uncertainty in the map frame
float32 sigma_yaw_rad    # current 1-sigma heading uncertainty

geometry_msgs/PoseWithCovariance map_to_odom  # the correction currently applied
int32 last_tag_id        # id of the tag that produced the last fix (-1 = none)
float32 last_fix_xy_m    # metres of correction applied by the last fix
float32 last_fix_yaw_rad
