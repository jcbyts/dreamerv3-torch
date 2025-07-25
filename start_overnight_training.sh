#!/bin/bash

# VizDoom Overnight Training Launcher
# This script starts the training in a detached screen session
# so it continues running even if you disconnect
# Usage: ./start_overnight_training.sh

SESSION_NAME="vizdoom_training_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "VizDoom Overnight Training Launcher"
echo "=========================================="

# Check if screen is available
if ! command -v screen &> /dev/null; then
    echo "❌ Error: 'screen' is not installed"
    echo "Install with: sudo apt-get install screen"
    exit 1
fi

# Check if training script exists
if [ ! -f "./train_vizdoom_overnight.sh" ]; then
    echo "❌ Error: train_vizdoom_overnight.sh not found"
    echo "Make sure you're in the dreamerv3-torch directory"
    exit 1
fi

echo "🚀 Starting overnight training in screen session: $SESSION_NAME"
echo ""
echo "The training will run in the background and continue even if you:"
echo "- Close your terminal"
echo "- Disconnect from SSH"
echo "- Log out"
echo ""

# Start training in detached screen session
screen -dmS "$SESSION_NAME" bash -c "
    echo 'Starting VizDoom training at: \$(date)'
    echo 'Session: $SESSION_NAME'
    echo 'Working directory: \$(pwd)'
    echo ''
    ./train_vizdoom_overnight.sh
    echo ''
    echo 'Training completed at: \$(date)'
    echo 'Press any key to exit...'
    read -n 1
"

# Wait a moment for screen to start
sleep 2

# Check if session started successfully
if screen -list | grep -q "$SESSION_NAME"; then
    echo "✅ Training started successfully!"
    echo ""
    echo "📋 Session Information:"
    echo "  Session name: $SESSION_NAME"
    echo "  Status: Running in background"
    echo ""
    echo "🔍 To monitor progress:"
    echo "  ./monitor_training.sh"
    echo ""
    echo "📺 To attach to the training session:"
    echo "  screen -r $SESSION_NAME"
    echo ""
    echo "📊 To view TensorBoard (in another terminal):"
    echo "  tensorboard --logdir ./logdir --port 6006"
    echo "  Then open: http://localhost:6006"
    echo ""
    echo "⏹️  To stop training (if needed):"
    echo "  screen -S $SESSION_NAME -X quit"
    echo "  # or"
    echo "  pkill -f 'python.*dreamer.py'"
    echo ""
    echo "📋 To list all screen sessions:"
    echo "  screen -list"
    echo ""
    echo "🎯 Expected Training Time: 12-16 hours"
    echo "🎯 Scenarios: deadly_corridor, defend_center, health_gathering, basic"
    echo "🎯 Hardware: 2x RTX 6000 Ada Generation"
    echo ""
    echo "Training is now running in the background!"
    echo "You can safely close this terminal."
else
    echo "❌ Failed to start training session"
    echo "Check the error messages above"
    exit 1
fi
