#!/bin/bash

# Test script for VizDoom agent evaluation
# This script activates the conda environment and runs evaluation

echo "Testing VizDoom Agent Evaluation"
echo "================================"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dreamer

# Check if environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "dreamer" ]]; then
    echo "Error: Failed to activate dreamer environment"
    exit 1
fi

echo "✓ Activated conda environment: $CONDA_DEFAULT_ENV"

# List available models
echo ""
echo "Available trained models:"
python run_evaluation.py --list-models

# Test basic evaluation on vizdoom_basic_test
echo ""
echo "Testing basic evaluation..."
python evaluate_agent.py \
    --logdir logdir/vizdoom_basic_test \
    --config vizdoom_basic \
    --episodes 3 \
    --training-analysis

echo ""
echo "Testing video prediction..."
python evaluate_agent.py \
    --logdir logdir/vizdoom_basic_test \
    --config vizdoom_basic \
    --episodes 1 \
    --video-pred

echo ""
echo "Evaluation test complete!"
