#!/usr/bin/env bash
# tools/capture_dataset.sh
#
# Capture camera frames from the rover camera for K seeds.
# Runs headless and saves to ~/cat/dataset/seed_X/
#
# Usage:
#   bash tools/capture_dataset.sh --seeds 5 --frames 10 --output ~/cat/dataset
#   bash tools/capture_dataset.sh --seeds 5 --frames 10 --output ~/cat/dataset --headless

set -euo pipefail

# Defaults
NUM_SEEDS=5
FRAMES_PER_SEED=10
OUTPUT_DIR="$HOME/cat/dataset"
HEADLESS=true
SPAWN_DELAY=10.0
CAMERAS="cam_nav,cam_tag,cam_excavation,cam_dump"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)
            NUM_SEEDS="$2"
            shift 2
            ;;
        --frames)
            FRAMES_PER_SEED="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --headless)
            HEADLESS=true
            shift
            ;;
        --no-headless)
            HEADLESS=false
            shift
            ;;
        --spawn-delay)
            SPAWN_DELAY="$2"
            shift 2
            ;;
        --cameras)
            CAMERAS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== Dataset Capture ==="
echo "Seeds: $NUM_SEEDS"
echo "Frames per seed: $FRAMES_PER_SEED"
echo "Output dir: $OUTPUT_DIR"
echo "Headless: $HEADLESS"
echo "Cameras: $CAMERAS"
echo ""

mkdir -p "$OUTPUT_DIR"

# Source ROS
source /opt/ros/humble/setup.bash
if [[ -f "$HOME/cat/install/setup.bash" ]]; then
    source "$HOME/cat/install/setup.bash"
fi

# Function to capture frames for one seed
capture_seed() {
    local seed=$1
    local seed_dir="$OUTPUT_DIR/seed_$seed"

    echo "--- Capturing seed $seed ---"
    mkdir -p "$seed_dir"

    # Generate world for this seed if not exists
    if [[ ! -f "$HOME/cat/worlds/seed_$seed/arena.sdf" ]]; then
        echo "Generating world for seed $seed..."
        python3 -m cat_arena.generate --seed "$seed" --output-dir "$HOME/cat/worlds" --no-validate
    fi

    # Launch simulation with rover
    echo "Launching simulation for seed $seed..."
    HEADLESS_FLAG=""
    if [[ "$HEADLESS" == "true" ]]; then
        HEADLESS_FLAG="headless:=true headless_rendering:=true"
    fi

    # Launch in background
    ros2 launch rover_description sim.launch.py \
        seed:="$seed" \
        $HEADLESS_FLAG \
        spawn_delay:="$SPAWN_DELAY" \
        verbose:=1 \
        validate_world:=false \
        > "$seed_dir/sim.log" 2>&1 &

    LAUNCH_PID=$!
    echo "Simulation launched (PID: $LAUNCH_PID)"

    # Wait for simulation to start and rover to spawn
    sleep "$SPAWN_DELAY"
    sleep 5  # extra time for sensors

    # Capture frames using gz topic
    echo "Capturing $FRAMES_PER_SEED frames from cameras..."
    for cam in ${CAMERAS//,/ }; do
        mkdir -p "$seed_dir/$cam"
        for i in $(seq 1 $FRAMES_PER_SEED); do
            # Use gz topic to get image
            timeout 5 gz topic -e -t "/$cam" -n 1 --output "$seed_dir/$cam/frame_${i}.png" 2>/dev/null || true
            sleep 0.1
        done
    done

    # Kill the launch
    kill $LAUNCH_PID 2>/dev/null || true
    sleep 2
    pkill -f "gz sim" 2>/dev/null || true

    echo "Seed $seed done. Frames saved to $seed_dir"
}

# Main loop
for seed in $(seq 1 $NUM_SEEDS); do
    capture_seed $seed
done

echo ""
echo "=== Dataset capture complete ==="
echo "Data saved to: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"