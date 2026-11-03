# The behaviour sequencer's state, published on `/mission/status`.
#
# Module 5 owns the run: it decides *when* to drive, dig and dump, and it drives
# module 4's camera phases through `/perception/mode`.  This message is the one
# place that says what part of the run the rover believes it is in.
#
# The phase names below are the vocabulary of the competition task: leave the
# start zone, cross the obstacle zone to the excavation zone, dig, haul the sand
# back across the arena to the construction zone, and build the berm.

std_msgs/Header header

uint8 PHASE_IDLE=0            # not started
uint8 PHASE_ACQUIRE=1         # find a wall tag so the map is anchored
uint8 PHASE_TRAVEL_TO_DIG=2   # drive to the next excavation pose
uint8 PHASE_EXCAVATE=3        # dig (delegated to module 6)
uint8 PHASE_TRAVEL_TO_DUMP=4  # haul to the next berm pose
uint8 PHASE_DUMP=5            # dump (delegated to module 6)
uint8 PHASE_RETURN=6          # leave the work area
uint8 PHASE_PARK=7            # stop somewhere safe and legal
uint8 PHASE_COMPLETE=8        # the planned cycles are done
uint8 PHASE_ABORTED=9         # stopped for a reason that needs a human

uint8 phase
string phase_name
bool running                  # the sequencer is stepping (not idle/complete/aborted)
uint32 cycles_completed       # dig+dump cycles finished
uint32 cycles_target          # cycles the run asked for
float32 cycle_progress        # 0..1 of the current work cycle

# Why the run is or is not progressing.  Never empty for ABORTED.
string detail

# --- the excavation delegation --------------------------------------------
# Module 6 answers WorkCycle on this action name.  Until it exists the sequencer
# reports `excavation_available = false` and holds instead of pretending.
string excavation_action
bool excavation_available

# --- what the sequencer asked perception for ------------------------------
string perception_mode
