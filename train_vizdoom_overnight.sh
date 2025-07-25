#!/bin/bash

# VizDoom Overnight Training Script for DreamerV3
# Optimized for 2x RTX 6000 Ada Generation GPUs
# Author: Augment Agent
# Usage: ./train_vizdoom_overnight.sh

set -e  # Exit on any error

# Configuration
CONDA_ENV="dreamer"
BASE_LOGDIR="./logdir"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# GPU Configuration
GPU0="cuda:0"
GPU1="cuda:1"

# Training Parameters (optimized for your hardware)
STEPS_FULL="2e6"      # 2M steps for full training
STEPS_MEDIUM="1e6"    # 1M steps for medium training
ENVS="8"              # 8 parallel environments
BATCH_SIZE="32"       # Large batch size for your GPUs
EVAL_EVERY="2e4"      # Evaluate every 20k steps
EVAL_EPISODES="20"    # More evaluation episodes

echo "=========================================="
echo "VizDoom DreamerV3 Overnight Training"
echo "Started at: $(date)"
echo "Hardware: 2x RTX 6000 Ada Generation"
echo "=========================================="

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Check GPU availability
echo "Checking GPU availability..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Function to run training with error handling
run_training() {
    local scenario=$1
    local config=$2
    local device=$3
    local steps=$4
    local logdir_suffix=$5
    
    local logdir="${BASE_LOGDIR}/vizdoom_${scenario}_${TIMESTAMP}${logdir_suffix}"
    
    echo ""
    echo "=========================================="
    echo "Training: $scenario"
    echo "Config: $config"
    echo "Device: $device"
    echo "Steps: $steps"
    echo "Logdir: $logdir"
    echo "Started at: $(date)"
    echo "=========================================="
    
    # Create logdir
    mkdir -p "$logdir"
    
    # Save training parameters to logdir
    cat > "$logdir/training_params.txt" << EOF
Scenario: $scenario
Config: $config
Device: $device
Steps: $steps
Envs: $ENVS
Batch Size: $BATCH_SIZE
Eval Every: $EVAL_EVERY
Eval Episodes: $EVAL_EPISODES
Started: $(date)
Hardware: 2x RTX 6000 Ada Generation
EOF
    
    # Run training with comprehensive logging
    python dreamer.py \
        --configs "$config" \
        --task "vizdoom_$scenario" \
        --logdir "$logdir" \
        --device "$device" \
        --steps "$steps" \
        --envs "$ENVS" \
        --batch_size "$BATCH_SIZE" \
        --eval_every "$EVAL_EVERY" \
        --eval_episode_num "$EVAL_EPISODES" \
        --parallel False \
        --video_pred_log True \
        2>&1 | tee "$logdir/training.log"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ SUCCESS: $scenario training completed at $(date)"
        echo "$(date): SUCCESS - $scenario" >> "${BASE_LOGDIR}/training_summary_${TIMESTAMP}.log"
    else
        echo "❌ FAILED: $scenario training failed with exit code $exit_code at $(date)"
        echo "$(date): FAILED - $scenario (exit code: $exit_code)" >> "${BASE_LOGDIR}/training_summary_${TIMESTAMP}.log"
    fi
    
    return $exit_code
}

# Create base logdir
mkdir -p "$BASE_LOGDIR"

# Initialize summary log
echo "VizDoom DreamerV3 Training Summary - Started: $(date)" > "${BASE_LOGDIR}/training_summary_${TIMESTAMP}.log"

echo ""
echo "🚀 Starting overnight training sequence..."
echo "This will train on the most challenging VizDoom scenarios"
echo "Estimated total time: 12-16 hours"
echo ""

# Training Sequence - Most Challenging Scenarios First
# Using both GPUs for maximum throughput

echo "Phase 1: Deadly Corridor (Most Challenging) - GPU 0"
echo "This scenario requires navigation, combat, and survival skills"
if run_training "deadly_corridor" "vizdoom_deadly_corridor" "$GPU0" "$STEPS_FULL" "_phase1"; then
    echo "✅ Phase 1 completed successfully"
else
    echo "❌ Phase 1 failed, but continuing..."
fi

echo ""
echo "Phase 2: Defend the Center (Combat Focus) - GPU 1"
echo "This scenario focuses on combat and target tracking"
if run_training "defend_the_center" "vizdoom_defend_center" "$GPU1" "$STEPS_MEDIUM" "_phase2"; then
    echo "✅ Phase 2 completed successfully"
else
    echo "❌ Phase 2 failed, but continuing..."
fi

echo ""
echo "Phase 3: Health Gathering (Navigation + Survival) - GPU 0"
echo "This scenario requires efficient navigation and resource management"
if run_training "health_gathering" "vizdoom_health_gathering" "$GPU0" "$STEPS_MEDIUM" "_phase3"; then
    echo "✅ Phase 3 completed successfully"
else
    echo "❌ Phase 3 failed, but continuing..."
fi

echo ""
echo "Phase 4: Basic Scenario (Baseline) - GPU 1"
echo "Simple scenario for comparison and validation"
if run_training "basic" "vizdoom_basic" "$GPU1" "$STEPS_MEDIUM" "_phase4"; then
    echo "✅ Phase 4 completed successfully"
else
    echo "❌ Phase 4 failed"
fi

# Final summary
echo ""
echo "=========================================="
echo "🏁 OVERNIGHT TRAINING COMPLETED"
echo "Finished at: $(date)"
echo "=========================================="

echo ""
echo "📊 Training Summary:"
cat "${BASE_LOGDIR}/training_summary_${TIMESTAMP}.log"

echo ""
echo "📁 Results Location:"
echo "All training results are saved in: ${BASE_LOGDIR}/"
echo "Logs with timestamp: ${TIMESTAMP}"

echo ""
echo "🔍 To view results:"
echo "tensorboard --logdir ${BASE_LOGDIR} --port 6006"
echo "Then open: http://localhost:6006"

echo ""
echo "📈 Key metrics to watch in TensorBoard:"
echo "- eval_return (should increase over time)"
echo "- eval_length (episode length)"
echo "- model_loss (should decrease)"
echo "- actor_loss (policy learning)"
echo "- Videos in 'Images' tab"

echo ""
echo "🎯 Expected Results:"
echo "- Deadly Corridor: Most challenging, expect gradual improvement"
echo "- Defend Center: Should learn to aim and shoot effectively"
echo "- Health Gathering: Should learn efficient navigation"
echo "- Basic: Should achieve good performance quickly"

echo ""
echo "Training completed! Check the logs and TensorBoard for detailed results."
