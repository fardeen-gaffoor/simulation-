# How much of the berm target has been built up to now (module 6).
#
# The berm is sand, and the simulator has no sand: this message *is* the berm as
# far as the software is concerned.  The planner in module 5 keeps
# `arena.zones.berm_target` as a keep-out from the moment the first dump lands
# (it latches /mission/berm_started itself), so the navigator and this model
# agree about where the berm is by construction rather than by measurement.

std_msgs/Header header

float32 target_volume_m3   # berm_target area x target height
float32 placed_volume_m3   # sand that landed inside the target footprint
float32 spilled_volume_m3  # sand that missed it - the quality number
float32 completion         # 0..1, min(1, placed / target): the judged volume score
                           # (coverage is the separate shape metric; a flat prism
                           # at target height would need ~1.4x the target volume)

float32 mean_height_m
float32 peak_height_m
float32 coverage           # fraction of target cells at/above the target height
uint32 cells_total
uint32 cells_filled

float32 target_height_m
uint32 dumps               # work cycles that deposited sand

string message
