# What the velocity gate actually let through, published on `/nav/gate_status`.
#
# The gate is the single writer of `/cmd_vel`.  It exists so that there is
# exactly one place where the rover can be stopped, and so that "why is the
# rover not moving" has one answer.  `command` is the twist that was published.

std_msgs/Header header

bool estop                   # the /estop contract is latched
bool allowed                 # the incoming command was forwarded (possibly scaled)
bool stale_command           # no fresh command arrived within the timeout
bool emergency_brake         # an obstacle appeared inside the reactive stop distance

# Localisation gating: the map is only trustworthy while the tags anchor it.
bool localized
float32 speed_scale          # 0..1, scaled by the localisation uncertainty
float32 sigma_xy_m
string reason                # "" when allowed at full scale; otherwise why the command
                             # was zeroed, or which factor scaled it (see speed_scale)

geometry_msgs/Twist command  # what was published on /cmd_vel

# The limits in force for this tick (after localisation scaling).
float32 max_speed_mps
float32 max_yaw_rate_radps
