#!/bin/bash

# Last training run: 1M steps, 3 seeds, health pickup + deadly corridor scenarios
# This will run the full hierarchical model comparison on two meaningful VizDoom tasks

echo "Starting 1M step training comparison on health pickup and deadly corridor scenarios..."
echo "This will take many hours to complete."

# Health pickup scenario on CUDA:0
echo "Starting health pickup scenario training on CUDA:0..."
./compare_models.sh --steps 1000000 --seeds 3 --task vizdoom_health_gathering --device cuda:0 &

# Wait a moment to avoid simultaneous startup issues
sleep 10

# Deadly corridor scenario on CUDA:1  
echo "Starting deadly corridor scenario training on CUDA:1..."
./compare_models.sh --steps 1000000 --seeds 3 --task vizdoom_deadly_corridor --device cuda:1 &

echo "Both training runs started in background."
echo "Monitor progress with:"
echo "  tensorboard --logdir ./comparison_results"
echo ""
echo "Check GPU usage with:"
echo "  nvidia-smi"
echo ""
echo "Training will complete in several hours."
echo "Results will be in ./comparison_results/"

# Wait for both background jobs to complete
wait

echo "All training runs completed!"
echo "Check results in ./comparison_results/"
