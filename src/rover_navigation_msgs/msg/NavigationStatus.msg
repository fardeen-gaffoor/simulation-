# The live state of the navigation node, published on `/nav/status`.
#
# Module 5 is the only part of the stack that decides how fast the rover drives,
# so this message is the operator's (and the judges') window into that decision:
# what it is doing, where it thinks it is, how far is left, and - when it is not
# driving - exactly why not.
#
# `header.frame_id` is the map frame; `current_pose` is the rover in that frame.

std_msgs/Header header

# --- state machine ---------------------------------------------------------
# The same numbers are reported in NavigateTo feedback, so a client can watch a
# goal without subscribing to this topic.
uint8 STATE_IDLE=0        # no goal; a stopped rover that is still ready
uint8 STATE_PLANNING=1    # a goal was accepted and a path is being searched
uint8 STATE_TURNING=2     # rotating in place toward the path
uint8 STATE_DRIVING=3     # following the path
uint8 STATE_ARRIVED=4     # at the goal (position and heading within tolerance)
uint8 STATE_HOLDING=5     # stopped on purpose (not localised, not allowed, estop)
uint8 STATE_BLOCKED=6     # no safe path right now; will re-plan and retry
uint8 STATE_FAILED=7      # the goal cannot be achieved; a reason is in `message`

uint8 state
string state_name

bool active               # a goal is currently accepted and being executed
bool localized            # the localizer reports map -> odom is anchored
bool map_ready            # at least one obstacle frame has been folded into the map

# --- goal ------------------------------------------------------------------
# Empty when no goal has ever been accepted.
string goal_id
geometry_msgs/PoseStamped goal_pose
geometry_msgs/PoseStamped current_pose
float32 distance_remaining_m
float32 path_length_m

# --- the command that leaves this node ------------------------------------
float32 speed_mps
float32 yaw_rate_radps

# --- map / anchoring health (copied from LocalizationState so a judge sees it
#     without subscribing to module 4) ---------------------------------------
float32 sigma_xy_m
float32 sigma_yaw_rad

# --- bookkeeping -----------------------------------------------------------
uint32 replans                # planner runs since the goal was accepted
uint32 obstacles_in_map       # obstacles remembered in the costmap
uint32 blocked_count          # times the planner found no path for this goal
uint32 goal_rejections        # goals refused (outside arena, inside an obstacle, ...)
string message                # human-readable detail; "" when nothing to say
