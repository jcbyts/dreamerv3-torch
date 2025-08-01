#!/usr/bin/env python3
"""
Comprehensive evaluation script for DreamerV3 vs hDreamer comparison
Generates:
1. Linear probe analysis with bar plots (PNG)
2. World model dreams (MP4)
3. Agent behavior videos (MP4)
"""

import argparse
import pathlib
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import torch

# Add current directory to path
sys.path.append(str(pathlib.Path(__file__).parent))

def find_comparison_models(comparison_dir):
    """Find all models in a comparison directory."""
    comparison_path = pathlib.Path(comparison_dir)
    
    models = {
        'dreamerv3': [],
        'hdreamer': []
    }
    
    for model_dir in comparison_path.glob("*_seed_*"):
        if model_dir.is_dir() and (model_dir / "latest.pt").exists():
            if 'dreamerv3' in model_dir.name:
                models['dreamerv3'].append(model_dir)
            elif 'hdreamer' in model_dir.name:
                models['hdreamer'].append(model_dir)
    
    # Sort by seed number
    for model_type in models:
        models[model_type].sort(key=lambda x: int(x.name.split('_seed_')[1]))
    
    return models

def run_linear_probe_analysis(models, output_dir, config_name, device='cuda:0'):
    """Run linear probe analysis for all models and generate comparison plots."""
    print("🧠 Running linear probe analysis...")
    
    from eval_ego_linear_probe import run_ego_probe
    
    results = defaultdict(list)
    
    # Run linear probing for each model
    for model_type, model_list in models.items():
        for i, model_dir in enumerate(model_list):
            print(f"  Analyzing {model_type} seed {i+1}...")
            
            try:
                # Run linear probing
                probe_results = run_ego_probe(
                    model_path=str(model_dir),
                    config_name=config_name,
                    episodes_train=10,
                    episodes_test=5,
                    max_steps_per_episode=300,
                    device=device
                )

                if probe_results:
                    results[model_type].append(probe_results)
                    print(f"    ✅ R² total: {probe_results.get('r2_total', 0):.3f}")
                    print(f"       Position: {probe_results.get('r2_position', 0):.3f}")
                    print(f"       Velocity: {probe_results.get('r2_velocity', 0):.3f}")
                    print(f"       Angle: {probe_results.get('r2_angle', 0):.3f}")
                else:
                    print(f"    ❌ Failed")
                    
            except Exception as e:
                import traceback
                print(f"    ❌ Error: {e}")
                print(f"    Full traceback: {traceback.format_exc()}")
    
    # Generate comparison plots
    if results:
        plot_linear_probe_comparison(results, output_dir)
    
    return results

def plot_linear_probe_comparison(results, output_dir):
    """Generate bar plots comparing linear probe results."""
    print("📊 Generating linear probe comparison plots...")
    
    # Set up colors
    colors = {
        'dreamerv3': '#2E86AB',  # Blue
        'hdreamer': '#A23B72'    # Purple
    }
    
    # Extract R² scores
    r2_data = []
    for model_type, model_results in results.items():
        for i, result in enumerate(model_results):
            r2_data.append({
                'Model': f"{model_type.title()}",
                'Seed': i + 1,
                'R² Total': result.get('r2_total', 0),
                'R² Position': result.get('r2_position', 0),
                'R² Velocity': result.get('r2_velocity', 0),
                'R² Angle': result.get('r2_angle', 0)
            })
    
    if not r2_data:
        print("  ❌ No data to plot")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Linear Probe Analysis: DreamerV3 vs hDreamer', fontsize=16, fontweight='bold')
    
    metrics = ['R² Total', 'R² Position', 'R² Velocity', 'R² Angle']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Prepare data for plotting
        dreamer_scores = [d[metric] for d in r2_data if 'DreamerV3' in d['Model']]
        hdreamer_scores = [d[metric] for d in r2_data if 'Hdreamer' in d['Model']]
        
        x_pos = np.arange(max(len(dreamer_scores), len(hdreamer_scores)))
        width = 0.35
        
        # Plot bars
        if dreamer_scores:
            ax.bar(x_pos - width/2, dreamer_scores, width, 
                  label='DreamerV3', color=colors['dreamerv3'], alpha=0.8)
        
        if hdreamer_scores:
            ax.bar(x_pos + width/2, hdreamer_scores, width,
                  label='hDreamer', color=colors['hdreamer'], alpha=0.8)
        
        ax.set_title(metric, fontweight='bold')
        ax.set_xlabel('Seed')
        ax.set_ylabel('R² Score')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Seed {i+1}' for i in range(len(x_pos))])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'linear_probe_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved comparison plot: {plot_path}")

def generate_videos(models, output_dir, config_name, device='cuda:0'):
    """Generate world model dreams and agent behavior videos."""
    print("🎬 Generating videos...")
    
    from evaluate_agent import AgentEvaluator
    
    videos_dir = output_dir / 'videos'
    videos_dir.mkdir(exist_ok=True)
    
    for model_type, model_list in models.items():
        for i, model_dir in enumerate(model_list):
            seed_name = f"{model_type}_seed_{i+1}"
            print(f"  Generating videos for {seed_name}...")
            
            try:
                evaluator = AgentEvaluator(str(model_dir), config_name, device)
                
                # Generate world model dreams
                print(f"    🌙 World model dreams...")
                predictions = evaluator.generate_video_predictions(num_sequences=3)

                if predictions:
                    dream_path = videos_dir / f'{seed_name}_dreams.mp4'
                    evaluator.create_comparison_video(predictions, str(dream_path))
                    print(f"      ✅ Saved: {dream_path}")
                else:
                    print(f"      ❌ No predictions generated")

                # Generate agent performance evaluation (includes behavior analysis)
                print(f"    🎮 Agent performance...")
                performance_results = evaluator.evaluate_performance(num_episodes=3)

                if performance_results:
                    print(f"      ✅ Performance: {performance_results['metrics'].get('eval_return', 'N/A'):.2f} return")
                else:
                    print(f"      ❌ Performance evaluation failed")
                    
            except Exception as e:
                print(f"    ❌ Error generating videos: {e}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of DreamerV3 vs hDreamer')
    parser.add_argument('--comparison-dir', type=str, required=True,
                       help='Path to comparison directory (e.g., ./logdir/comparison_20250727_114209)')
    parser.add_argument('--results-base-dir', type=str, default='./results',
                       help='Base directory for results (default: ./results)')
    parser.add_argument('--config', type=str, default='vizdoom_health_gathering',
                       help='Config name for the task')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--skip-linear', action='store_true',
                       help='Skip linear probe analysis')
    parser.add_argument('--skip-videos', action='store_true',
                       help='Skip video generation')

    args = parser.parse_args()

    print("🔬 Comprehensive Model Evaluation")
    print("=" * 50)

    # Setup directories - extract comparison name and create results subdirectory
    comparison_dir = pathlib.Path(args.comparison_dir)
    comparison_name = comparison_dir.name  # e.g., "comparison_20250727_114209"

    results_base_dir = pathlib.Path(args.results_base_dir)
    output_dir = results_base_dir / comparison_name

    print(f"📁 Comparison: {comparison_name}")
    print(f"📂 Results will be saved to: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not comparison_dir.exists():
        print(f"❌ Comparison directory not found: {comparison_dir}")
        return
    
    # Find models
    models = find_comparison_models(comparison_dir)
    
    print(f"📁 Found models:")
    for model_type, model_list in models.items():
        print(f"  {model_type}: {len(model_list)} seeds")
    
    if not any(models.values()):
        print("❌ No trained models found")
        return
    
    # Run evaluations
    if not args.skip_linear:
        linear_results = run_linear_probe_analysis(models, output_dir, args.config, args.device)
    
    if not args.skip_videos:
        generate_videos(models, output_dir, args.config, args.device)
    
    print(f"\n✅ Evaluation complete!")
    print(f"📂 Results saved to: {output_dir}")
    print(f"📊 Linear probe plots: {output_dir}/linear_probe_comparison.png")
    print(f"🎬 Videos: {output_dir}/videos/")
    print(f"🔍 View results: ls -la {output_dir}")

if __name__ == "__main__":
    main()
