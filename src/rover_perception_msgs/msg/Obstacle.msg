# One confidence-weighted obstacle in the map (= arena) frame.
#
# The competition forbids giving the autonomy a precomputed obstacle list, so
# every obstacle here comes from live camera perception and carries the evidence:
# how many frames saw it, when it was last seen and how confident the tracker is.
# Craters are reported with a deliberately conservative bias (see
# docs/detector.md): the cost of treating sand as a hole is one cautious detour,
# the cost of driving into a hole is a failed run.

std_msgs/Header header

# Obstacle classes.  The numeric values are a wire contract with module 5 and
# with the YOLO class order of module 3 (0 = boulder, 1 = crater).
uint8 KIND_UNKNOWN=0
uint8 KIND_BOULDER=1
uint8 KIND_CRATER=2

string id            # stable track id, e.g. "boulder_3" (unique per run)
uint8 kind           # KIND_* above
uint8 class_id       # 0 = boulder, 1 = crater (module 3's YOLO class index; 255 = n/a)

# Pose of the obstacle's footprint centre in the map frame: x, y on the sand
# plane, z = 0 for a crater and the boulder's exposed top height for a boulder.
# orientation is the identity (obstacles are round); the covariance is populated
# with the perception uncertainty (m^2) so a consumer can weight the evidence.
geometry_msgs/PoseWithCovariance pose

float32 radius       # m, footprint radius in the sand plane
float32 height       # m, boulder exposed height above sand (0 for craters)
float32 depth        # m, crater depth below sand, positive down (0 for boulders)

float32 confidence   # 0..1, tracker confidence (not a raw per-frame score)
float32 score        # best raw detector score seen for this track (0..1)
uint32 observations  # frames that contributed evidence
float32 age_s        # seconds since the track was created
float32 last_seen_age_s  # seconds since the last contributing frame
string source        # detector backend that produced it ("classical", "onnx", ...)
bool conservative    # true when the hit was inflated for safety (e.g. ambiguous crater)
