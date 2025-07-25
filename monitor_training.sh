#!/bin/bash

# VizDoom Training Monitor Script
# Use this to check training progress while it's running
# Usage: ./monitor_training.sh

LOGDIR="./logdir"

echo "=========================================="
echo "VizDoom Training Monitor"
echo "Current time: $(date)"
echo "=========================================="

# Check if training is running
echo "🔍 Checking for running training processes..."
DREAMER_PROCS=$(pgrep -f "python.*dreamer.py" | wc -l)
if [ $DREAMER_PROCS -gt 0 ]; then
    echo "✅ Found $DREAMER_PROCS active training process(es)"
    echo ""
    echo "Running processes:"
    ps aux | grep "python.*dreamer.py" | grep -v grep
else
    echo "❌ No active training processes found"
fi

echo ""
echo "=========================================="

# Check GPU usage
echo "🖥️  GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
    while IFS=, read -r idx name util mem_used mem_total temp; do
        echo "GPU $idx ($name): ${util}% util, ${mem_used}MB/${mem_total}MB memory, ${temp}°C"
    done
else
    echo "nvidia-smi not available"
fi

echo ""
echo "=========================================="

# Check recent log directories
echo "📁 Recent training runs:"
if [ -d "$LOGDIR" ]; then
    find "$LOGDIR" -maxdepth 1 -type d -name "vizdoom_*" -printf "%T@ %Tc %p\n" | sort -n | tail -5 | while read timestamp date time timezone path; do
        scenario=$(basename "$path" | sed 's/vizdoom_\([^_]*\)_.*/\1/')
        echo "  $scenario: $path (created: $date $time)"
    done
else
    echo "  No logdir found yet"
fi

echo ""
echo "=========================================="

# Check latest training logs
echo "📊 Latest training progress:"
LATEST_LOG=$(find "$LOGDIR" -name "training.log" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
    echo "Latest log: $LATEST_LOG"
    echo ""
    echo "Last 10 lines:"
    tail -10 "$LATEST_LOG"
    echo ""
    
    # Extract key metrics from log
    echo "🎯 Key Metrics (latest values):"
    
    # Extract eval_return
    EVAL_RETURN=$(grep "eval_return" "$LATEST_LOG" | tail -1 | sed 's/.*eval_return \([0-9.-]*\).*/\1/')
    if [ -n "$EVAL_RETURN" ]; then
        echo "  Latest eval_return: $EVAL_RETURN"
    fi
    
    # Extract eval_length
    EVAL_LENGTH=$(grep "eval_length" "$LATEST_LOG" | tail -1 | sed 's/.*eval_length \([0-9.-]*\).*/\1/')
    if [ -n "$EVAL_LENGTH" ]; then
        echo "  Latest eval_length: $EVAL_LENGTH"
    fi
    
    # Extract step count
    STEP_COUNT=$(grep -o '\[[0-9]*\]' "$LATEST_LOG" | tail -1 | tr -d '[]')
    if [ -n "$STEP_COUNT" ]; then
        echo "  Current step: $STEP_COUNT"
    fi
    
    # Count evaluations
    EVAL_COUNT=$(grep -c "Start evaluation" "$LATEST_LOG")
    echo "  Evaluations completed: $EVAL_COUNT"
    
else
    echo "No training logs found yet"
fi

echo ""
echo "=========================================="

# Check summary logs
echo "📋 Training Summary:"
SUMMARY_LOG=$(find "$LOGDIR" -name "training_summary_*.log" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

if [ -n "$SUMMARY_LOG" ] && [ -f "$SUMMARY_LOG" ]; then
    echo "Summary log: $SUMMARY_LOG"
    echo ""
    cat "$SUMMARY_LOG"
else
    echo "No summary log found yet"
fi

echo ""
echo "=========================================="

# Quick TensorBoard reminder
echo "🔗 To view detailed progress:"
echo "tensorboard --logdir $LOGDIR --port 6006"
echo "Then open: http://localhost:6006"

echo ""
echo "🔄 To run this monitor again:"
echo "./monitor_training.sh"

echo ""
echo "⏹️  To stop training (if needed):"
echo "pkill -f 'python.*dreamer.py'"

echo ""
echo "Monitor completed at: $(date)"
