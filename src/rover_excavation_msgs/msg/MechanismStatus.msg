# What the excavation mechanism is doing right now (module 6).
#
# Published by rover_excavation at `mechanism.report_hz`.  The ladder pitch and
# belt velocity here are the *commanded* actuators echoed from the dynamics
# model, so a bag file is self-contained; the physical joints are on
# /joint_states (bridged out of Gazebo, produced by the drive firmware on the
# real rover).
#
# The one field that matters for safety is `travel_ready`: the ladder must be
# raised back inside its travel attitude before the navigation module is allowed
# to drive anywhere, and module 5 does not know that by itself.

std_msgs/Header header

# STOWED     - ladder raised, belt stopped, gate shut: the travel attitude
# DEPLOYING  - ladder is slewing down toward the dig attitude
# DIGGING    - belt running forward, sand entering the tub
# DUMPING    - gate open, belt reversing, sand leaving the tub onto the berm
# RETRACTING - ladder is slewing back up to the travel attitude
# FAULT      - an interlock tripped; the mechanism was stopped and is held
uint8 STATE_STOWED=0
uint8 STATE_DEPLOYING=1
uint8 STATE_DIGGING=2
uint8 STATE_DUMPING=3
uint8 STATE_RETRACTING=4
uint8 STATE_FAULT=5

uint8 state
string state_name

# True when the ladder is inside the travel attitude and the belt is stopped.
# The rover may be driven only while this is true.
bool travel_ready

float32 ladder_pitch_rad      # + = tip down (matches the URDF convention)
float32 ladder_depth_m        # ladder tip below the sand surface, 0 if above it
float32 belt_velocity_radps   # signed: + digs, - dumps
float32 belt_linear_mps       # belt surface speed = velocity * roller radius
float32 tub_gate              # 0 = shut, 1 = fully open

float32 tub_fill_m3
float32 tub_capacity_m3
float32 tub_fraction          # 0..1

# One line for a human: what the mechanism is doing, or why it faulted.
string message
