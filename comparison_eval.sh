#!/bin/bash

# Comprehensive evaluation script for multiple comparison directories
# Runs linear probe analysis and video generation for DreamerV3 vs hDreamer comparisons

echo "🔬 Running Comprehensive Evaluation on Multiple Comparisons"
echo "=" * 60

# List of comparison directories to evaluate
COMPARISON_DIRS=(
    "./logdir/comparison_20250727_114209"  # Health gathering
    "./logdir/comparison_20250727_114219"  # Deadly corridor
)

# Configuration for each comparison (auto-detect from directory structure)
declare -A CONFIGS
CONFIGS["comparison_20250727_114209"]="vizdoom_health_gathering"
CONFIGS["comparison_20250727_114219"]="vizdoom_deadly_corridor"

# Results will be saved to ./results/
RESULTS_BASE_DIR="./results"

echo "📁 Comparisons to evaluate:"
for dir in "${COMPARISON_DIRS[@]}"; do
    comparison_name=$(basename "$dir")
    config=${CONFIGS[$comparison_name]}
    echo "  - $comparison_name (config: $config)"
done

echo ""
echo "🚀 Starting evaluations..."

# Track success/failure
SUCCESS_COUNT=0
TOTAL_COUNT=${#COMPARISON_DIRS[@]}

for comparison_dir in "${COMPARISON_DIRS[@]}"; do
    comparison_name=$(basename "$comparison_dir")
    config=${CONFIGS[$comparison_name]}
    
    echo ""
    echo "=" * 60
    echo "🔍 Evaluating: $comparison_name"
    echo "📂 Directory: $comparison_dir"
    echo "⚙️  Config: $config"
    echo "=" * 60
    
    # Check if directory exists
    if [ ! -d "$comparison_dir" ]; then
        echo "❌ Directory not found: $comparison_dir"
        echo "   Skipping..."
        continue
    fi
    
    # Check if models exist
    model_count=$(find "$comparison_dir" -name "latest.pt" | wc -l)
    if [ "$model_count" -eq 0 ]; then
        echo "❌ No trained models found in: $comparison_dir"
        echo "   Skipping..."
        continue
    fi
    
    echo "✅ Found $model_count trained models"
    
    # Run comprehensive evaluation
    echo "🏃 Running evaluation..."
    
    if python comprehensive_evaluation.py \
        --comparison-dir "$comparison_dir" \
        --config "$config" \
        --results-base-dir "$RESULTS_BASE_DIR" \
        --device cuda:0; then
        
        echo "✅ Evaluation completed successfully for $comparison_name"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        
        # Show results location
        results_dir="$RESULTS_BASE_DIR/$comparison_name"
        echo "📊 Results saved to: $results_dir"
        
        if [ -f "$results_dir/linear_probe_comparison.png" ]; then
            echo "   📈 Linear probe plot: $results_dir/linear_probe_comparison.png"
        fi
        
        if [ -d "$results_dir/videos" ]; then
            video_count=$(find "$results_dir/videos" -name "*.mp4" | wc -l)
            echo "   🎬 Videos ($video_count): $results_dir/videos/"
        fi
        
    else
        echo "❌ Evaluation failed for $comparison_name"
        echo "   Check the output above for error details"
    fi
done

echo ""
echo "=" * 60
echo "📋 Evaluation Summary"
echo "=" * 60
echo "✅ Successful: $SUCCESS_COUNT / $TOTAL_COUNT"

if [ "$SUCCESS_COUNT" -gt 0 ]; then
    echo ""
    echo "📂 All results saved to: $RESULTS_BASE_DIR/"
    echo "🔍 View results:"
    echo "   ls -la $RESULTS_BASE_DIR/"
    echo ""
    echo "📊 Linear probe comparisons:"
    find "$RESULTS_BASE_DIR" -name "linear_probe_comparison.png" -exec echo "   {}" \;
    echo ""
    echo "🎬 Video directories:"
    find "$RESULTS_BASE_DIR" -name "videos" -type d -exec echo "   {}" \;
fi

if [ "$SUCCESS_COUNT" -eq "$TOTAL_COUNT" ]; then
    echo ""
    echo "🎉 All evaluations completed successfully!"
else
    echo ""
    echo "⚠️  Some evaluations failed. Check the output above for details."
fi

echo ""
echo "🏁 Evaluation script complete."
