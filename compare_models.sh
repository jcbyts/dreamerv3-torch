#!/bin/bash

# DreamerV3 vs hDreamer Comparison Script
# Uses the existing training infrastructure with screen sessions
# 
# Usage:
#   ./compare_models.sh --steps 50000 --seeds 3
#   ./compare_models.sh --quick  # 5000 steps, 1 seed for testing

set -e

# Default parameters
STEPS=50000
SEEDS=3
TASK="vizdoom_basic"
DEVICE="cuda:0"
QUICK_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            STEPS=5000
            SEEDS=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --steps N     Training steps (default: 50000)"
            echo "  --seeds N     Number of seeds (default: 3)"
            echo "  --task NAME   Task name (default: vizdoom_basic)"
            echo "  --device DEV  Device (default: cuda:0)"
            echo "  --quick       Quick test: 5000 steps, 1 seed"
            echo "  -h, --help    Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Quick mode settings
if [ "$QUICK_MODE" = true ]; then
    echo "🚀 Quick comparison mode: $STEPS steps, $SEEDS seed(s)"
else
    echo "🚀 Full comparison mode: $STEPS steps, $SEEDS seed(s)"
fi

# Create experiment directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="./logdir/comparison_${TIMESTAMP}"
mkdir -p "$EXPERIMENT_DIR"

echo "📁 Experiment directory: $EXPERIMENT_DIR"
echo "📊 Task: $TASK"
echo "🖥️  Device: $DEVICE"
echo ""

# Save experiment config
cat > "$EXPERIMENT_DIR/experiment_config.yaml" << EOF
experiment_name: "dreamerv3_vs_hdreamer"
timestamp: "$TIMESTAMP"
parameters:
  steps: $STEPS
  seeds: $SEEDS
  task: "$TASK"
  device: "$DEVICE"
models:
  dreamerv3:
    configs: ["vizdoom", "tiny_dreamerv3"]
    description: "Tiny DreamerV3 with 12x32 flat latents (384 capacity)"
  hdreamer:
    configs: ["vizdoom", "tiny_hdreamer"] 
    description: "Tiny hDreamer with 4x32+8x32 hierarchical latents (384 capacity)"
EOF

echo "💾 Saved experiment config to $EXPERIMENT_DIR/experiment_config.yaml"
echo ""

# Function to start training in screen session
start_training() {
    local model_name=$1
    local configs=$2
    local seed=$3
    local session_name="comparison_${model_name}_seed${seed}_${TIMESTAMP}"
    local logdir="$EXPERIMENT_DIR/${model_name}_seed_${seed}"
    
    echo "🚀 Starting $model_name (seed $seed) in screen session: $session_name"
    
    # Start training in detached screen session
    screen -dmS "$session_name" bash -c "
        echo 'Starting $model_name training at: \$(date)'
        echo 'Session: $session_name'
        echo 'Logdir: $logdir'
        echo 'Configs: $configs'
        echo ''
        
        python dreamer.py \\
            --configs $configs \\
            --seed $seed \\
            --steps $STEPS \\
            --task $TASK \\
            --logdir $logdir \\
            --device $DEVICE
        
        echo ''
        echo '$model_name training completed at: \$(date)'
        echo 'Press any key to exit screen session...'
        read -n 1
    "
    
    # Wait a moment for screen to start
    sleep 1
    
    # Check if session started
    if screen -list | grep -q "$session_name"; then
        echo "✅ $model_name (seed $seed) started successfully"
        echo "   Session: $session_name"
        echo "   Logdir: $logdir"
    else
        echo "❌ Failed to start $model_name (seed $seed)"
        return 1
    fi
}

# Start all training sessions
echo "🎯 Starting training sessions..."
echo ""

SESSIONS=()

for seed in $(seq 1 $SEEDS); do
    echo "=== Seed $seed/$SEEDS ==="
    
    # Start DreamerV3
    start_training "dreamerv3" "vizdoom tiny_dreamerv3" $seed
    SESSIONS+=("comparison_dreamerv3_seed${seed}_${TIMESTAMP}")
    
    # Start hDreamer  
    start_training "hdreamer" "vizdoom tiny_hdreamer" $seed
    SESSIONS+=("comparison_hdreamer_seed${seed}_${TIMESTAMP}")
    
    echo ""
done

echo "🎉 All training sessions started!"
echo ""
echo "📋 Active Sessions:"
for session in "${SESSIONS[@]}"; do
    echo "  $session"
done
echo ""

# Create monitoring script for this experiment
cat > "$EXPERIMENT_DIR/monitor_comparison.sh" << 'EOF'
#!/bin/bash

# Monitor this specific comparison experiment

EXPERIMENT_DIR="$(dirname "$0")"
echo "=========================================="
echo "DreamerV3 vs hDreamer Comparison Monitor"
echo "Experiment: $(basename "$EXPERIMENT_DIR")"
echo "Time: $(date)"
echo "=========================================="

# Check running processes
echo "🔍 Active training sessions:"
screen -list | grep "comparison_.*$(basename "$EXPERIMENT_DIR" | cut -d'_' -f2)" || echo "  No active sessions found"
echo ""

# Check progress for each model
echo "📊 Training Progress:"
for model_dir in "$EXPERIMENT_DIR"/*/; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir")
        metrics_file="$model_dir/metrics.jsonl"
        
        if [ -f "$metrics_file" ]; then
            # Get latest metrics
            last_line=$(tail -1 "$metrics_file" 2>/dev/null)
            if [ -n "$last_line" ]; then
                step=$(echo "$last_line" | grep -o '"step":[0-9]*' | cut -d':' -f2)
                eval_return=$(echo "$last_line" | grep -o '"eval_return":[0-9.-]*' | cut -d':' -f2)
                
                if [ -n "$step" ] && [ -n "$eval_return" ]; then
                    echo "  $model_name: Step $step, Eval Return: $eval_return"
                else
                    echo "  $model_name: Step $step (no eval yet)"
                fi
            else
                echo "  $model_name: Starting..."
            fi
        else
            echo "  $model_name: No metrics yet"
        fi
    fi
done

echo ""
echo "🔗 TensorBoard: tensorboard --logdir $EXPERIMENT_DIR --port 6006"
echo "📺 Attach to session: screen -r comparison_MODEL_seedN_TIMESTAMP"
echo "📋 List sessions: screen -list"
echo "⏹️  Stop all: screen -list | grep comparison | cut -d. -f1 | xargs -I {} screen -S {} -X quit"
EOF

chmod +x "$EXPERIMENT_DIR/monitor_comparison.sh"

echo "🔍 Monitoring:"
echo "  ./logdir/comparison_${TIMESTAMP}/monitor_comparison.sh"
echo ""
echo "📊 TensorBoard:"
echo "  tensorboard --logdir $EXPERIMENT_DIR --port 6006"
echo "  Then open: http://localhost:6006"
echo ""
echo "📺 Attach to specific training:"
echo "  screen -r comparison_MODEL_seedN_${TIMESTAMP}"
echo ""
echo "📋 List all sessions:"
echo "  screen -list"
echo ""
echo "⏹️  Stop all comparison training:"
echo "  screen -list | grep comparison_.*_${TIMESTAMP} | cut -d. -f1 | xargs -I {} screen -S {} -X quit"
echo ""

if [ "$QUICK_MODE" = true ]; then
    echo "⏱️  Expected completion: ~20-30 minutes"
else
    echo "⏱️  Expected completion: ~2-3 hours"
fi

echo ""
echo "🎯 Training is now running in background screen sessions!"
echo "   You can safely close this terminal."
echo "   Use the monitoring commands above to check progress."
