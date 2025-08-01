#!/bin/bash

# Quick test to verify everything is ready for overnight training
# Usage: ./test_training_setup.sh

echo "=========================================="
echo "VizDoom Training Setup Test"
echo "=========================================="

# Test conda environment
echo "🔍 Testing conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dreamer

if [ $? -eq 0 ]; then
    echo "✅ Conda environment 'dreamer' activated successfully"
else
    echo "❌ Failed to activate conda environment 'dreamer'"
    exit 1
fi

# Test Python imports
echo ""
echo "🔍 Testing Python imports..."
python -c "
import torch
import vizdoom
from envs.vizdoom import ViZDoom
print('✅ All imports successful')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'VizDoom version: {vizdoom.__version__}')
"

if [ $? -ne 0 ]; then
    echo "❌ Python import test failed"
    exit 1
fi

# Test VizDoom environment creation
echo ""
echo "🔍 Testing VizDoom environment creation..."
python -c "
from envs.vizdoom import ViZDoom
env = ViZDoom('basic', size=(64, 64))
obs = env.reset()
print(f'✅ Environment created, image shape: {obs[\"image\"].shape}')
env.close()
"

if [ $? -ne 0 ]; then
    echo "❌ VizDoom environment test failed"
    exit 1
fi

# Test GPU availability
echo ""
echo "🔍 Testing GPU availability..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available with {torch.cuda.device_count()} GPU(s)')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('❌ CUDA not available')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ GPU test failed"
    exit 1
fi

# Test dreamer.py with minimal run
echo ""
echo "🔍 Testing DreamerV3 pipeline with minimal run..."
echo "This will run for 20 steps to verify everything works..."

python dreamer.py \
    --configs vizdoom_basic \
    --task vizdoom_basic \
    --logdir ./logdir/test_run \
    --steps 20 \
    --prefill 5 \
    --eval_every 10 \
    --eval_episode_num 1 \
    --envs 2

if [ $? -eq 0 ]; then
    echo "✅ DreamerV3 pipeline test successful!"
else
    echo "❌ DreamerV3 pipeline test failed"
    exit 1
fi

# Clean up test run
rm -rf ./logdir/test_run

echo ""
echo "=========================================="
echo "🎉 ALL TESTS PASSED!"
echo "=========================================="
echo ""
echo "Your system is ready for overnight training!"
echo ""
echo "🚀 To start overnight training:"
echo "  ./start_overnight_training.sh"
echo ""
echo "🔍 To monitor progress:"
echo "  ./monitor_training.sh"
echo ""
echo "📊 To view results:"
echo "  tensorboard --logdir ./logdir --port 6006"
echo ""
echo "Ready to train! 🚀"
