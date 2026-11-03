# The live obstacle map: everything module 5 is allowed to know about the
# obstacles in front of the rover.  header.frame_id is always the map frame.

std_msgs/Header header

Obstacle[] obstacles
uint32 frames_processed   # detector frames folded into this array
uint32 dropped_wall       # raw detections rejected by the arena-wall mask
uint32 dropped_range      # raw detections rejected as too close/far to trust
uint32 frames_without_fix # frames dropped because the map transform was unknown
