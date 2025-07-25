# VizDoom Agent Evaluation Guide

This guide explains how to use the comprehensive evaluation scripts for your trained VizDoom DreamerV3 agents.

## Overview

The evaluation system provides:
- **Performance Metrics**: Episode returns, lengths, success rates
- **Video Predictions**: World model imagination vs reality
- **Training Analysis**: Progress plots and learning curves  
- **Action Analysis**: Action distribution and behavior patterns
- **Comparison Videos**: Side-by-side truth vs model predictions

## Quick Start

### 1. List Available Models
```bash
python run_evaluation.py --list-models
```

### 2. Evaluate All Models (Basic)
```bash
python run_evaluation.py --episodes 10
```

### 3. Evaluate Specific Model
```bash
python run_evaluation.py --model deadly_corridor --episodes 20
```

### 4. Comprehensive Evaluation
```bash
python run_evaluation.py --model deadly_corridor --comprehensive
```

## Detailed Usage

### evaluate_agent.py

The main evaluation script with fine-grained control:

```bash
# Basic performance evaluation
python evaluate_agent.py --logdir logdir/vizdoom_deadly_corridor_20250721_143808_phase1 --episodes 10

# Generate video predictions
python evaluate_agent.py --logdir logdir/vizdoom_basic_test --video-pred

# Create comparison video (truth vs model)
python evaluate_agent.py --logdir logdir/vizdoom_basic_test --comparison-video

# Analyze training progress
python evaluate_agent.py --logdir logdir/vizdoom_basic_test --training-analysis

# Analyze action distribution
python evaluate_agent.py --logdir logdir/vizdoom_basic_test --action-analysis

# Run all analyses
python evaluate_agent.py --logdir logdir/vizdoom_basic_test --all
```

### Command Line Options

**evaluate_agent.py:**
- `--logdir`: Path to model checkpoint directory (required)
- `--config`: Configuration name (default: vizdoom)
- `--device`: Device to use - cuda/cpu (default: cuda)
- `--episodes`: Number of episodes to evaluate (default: 10)
- `--video-pred`: Generate video predictions
- `--comparison-video`: Create comparison video
- `--training-analysis`: Analyze training progress
- `--action-analysis`: Analyze action distribution
- `--all`: Run all analyses

**run_evaluation.py:**
- `--logdir`: Path to logdir containing models (default: logdir)
- `--model`: Specific model to evaluate (partial name match)
- `--episodes`: Number of episodes (default: 10)
- `--comprehensive`: Run all analyses
- `--list-models`: List available models

## Output Files

Each evaluation creates several output files in the model directory:

### Performance Metrics
- `evaluation_results.json`: Detailed performance metrics
- `evaluation_temp/`: Temporary evaluation episodes

### Visualizations
- `training_progress.png`: Training curves and metrics
- `action_distribution.png`: Action usage statistics
- `comparison_video.mp4`: Truth vs model prediction video
- `video_predictions.npz`: Raw video prediction data

### Example Output Structure
```
logdir/vizdoom_deadly_corridor_20250721_143808_phase1/
├── latest.pt                    # Trained model checkpoint
├── metrics.jsonl               # Training metrics
├── evaluation_results.json     # Evaluation results
├── training_progress.png       # Training analysis
├── action_distribution.png     # Action analysis
├── comparison_video.mp4        # Video comparison
├── video_predictions.npz       # Raw predictions
└── evaluation_temp/            # Temp evaluation data
```

## Understanding the Results

### Performance Metrics
- **Average Return**: Mean reward per episode
- **Average Length**: Mean episode duration
- **Success Rate**: Task-specific success metrics

### Video Predictions
The world model generates predictions showing:
- **Ground Truth**: Actual environment frames
- **Model Prediction**: What the world model imagines
- **Prediction Error**: Difference between truth and prediction

### Training Analysis
- **Training Return**: Learning progress over time
- **Evaluation Return**: Performance on held-out episodes
- **Episode Length**: How long episodes last
- **World Model Loss**: How well the model learns dynamics

### Action Analysis
- **Action Distribution**: Which actions the agent prefers
- **Action Frequency**: How often each action is used
- **Behavioral Patterns**: Agent's strategy insights

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   python evaluate_agent.py --device cpu --logdir your_model
   ```

2. **Missing ffmpeg for videos**
   ```bash
   sudo apt install ffmpeg  # Ubuntu/Debian
   brew install ffmpeg      # macOS
   ```

3. **Model not found**
   ```bash
   python run_evaluation.py --list-models
   ```

4. **Import errors**
   ```bash
   pip install matplotlib opencv-python
   ```

### Performance Tips

- Use `--episodes 5` for quick testing
- Use `--episodes 50` for thorough evaluation
- GPU evaluation is much faster than CPU
- Video generation requires significant memory

## Example Workflows

### Quick Model Check
```bash
# Fast check of model performance
python run_evaluation.py --model deadly_corridor --episodes 5
```

### Research Analysis
```bash
# Comprehensive analysis for paper/report
python evaluate_agent.py --logdir logdir/your_best_model --all --episodes 50
```

### Debugging Training
```bash
# Check if training is working
python evaluate_agent.py --logdir logdir/current_model --training-analysis
```

### Model Comparison
```bash
# Evaluate multiple models
python run_evaluation.py --comprehensive
```

## Integration with Existing Code

The evaluation scripts integrate seamlessly with your existing training:

- Uses the same configuration system (`configs.yaml`)
- Loads models using the same checkpoint format
- Leverages existing environment and dataset code
- Compatible with all VizDoom scenarios

The scripts are designed to work with your current training setup without requiring any changes to your existing code.
