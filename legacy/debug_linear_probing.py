#!/usr/bin/env python3
"""
Debug linear probing by plotting ground truth vs decoded values over time
"""

import argparse
import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import json

from enhanced_vizdoom import EnhancedViZDoom, extract_ground_truth_factors
from evaluate_agent import AgentEvaluator


class LinearProbingDebugger:
    def __init__(self, model_path, config_name, device='cuda'):
        self.model_path = pathlib.Path(model_path)
        self.device = device
        
        # Load the trained agent
        self.evaluator = AgentEvaluator(model_path, config_name, device)
        
        # Create enhanced environment with ground truth
        task = config_name.replace('vizdoom_', '')
        self.env = EnhancedViZDoom(task, enable_labels=True, enable_depth=True)
        
        print(f"Initialized debugging for {task}")
        
    def collect_sequence_data(self, num_episodes=5, max_steps_per_episode=50):
        """Collect latent representations and ground truth factors with sequence structure."""
        print(f"Collecting sequence data from {num_episodes} episodes...")
        
        sequences = []
        
        for episode in range(num_episodes):
            print(f"Episode {episode+1}/{num_episodes}")
            
            obs = self.env.reset()
            done = False
            step_count = 0
            
            # Initialize agent state
            agent_state = None
            
            episode_latents = []
            episode_factors = []
            episode_steps = []
            
            while not done and step_count < max_steps_per_episode:
                # Get latent representation from world model
                obs_tensor = {
                    'image': torch.tensor(obs['image']).unsqueeze(0).to(self.device),
                    'is_first': torch.tensor([step_count == 0]).to(self.device),
                    'is_last': torch.tensor([False]).to(self.device),
                    'is_terminal': torch.tensor([False]).to(self.device)
                }
                
                with torch.no_grad():
                    # Get latent state from world model
                    obs_processed = self.evaluator.agent._wm.preprocess(obs_tensor)
                    embed = self.evaluator.agent._wm.encoder(obs_processed)
                    
                    if agent_state is None:
                        latent = action = None
                    else:
                        latent, action = agent_state
                    
                    latent, _ = self.evaluator.agent._wm.dynamics.obs_step(
                        latent, action, embed, obs_tensor["is_first"]
                    )
                    
                    # Get feature representation
                    feat = self.evaluator.agent._wm.dynamics.get_feat(latent)
                    
                    # Get action for next step
                    policy_output, new_agent_state = self.evaluator.agent(
                        obs_tensor, obs_tensor["is_first"], agent_state, training=False
                    )
                    action = policy_output["action"]
                    agent_state = new_agent_state
                
                # Extract latent vector
                latent_vector = feat.cpu().numpy().flatten()
                episode_latents.append(latent_vector)
                
                # Extract ground truth factors
                gt_factors = extract_ground_truth_factors(obs)
                episode_factors.append(gt_factors)
                episode_steps.append(step_count)
                
                # Step environment
                action_np = action.cpu().numpy()
                if action_np.ndim > 0:
                    action_idx = int(action_np.flatten()[0])
                else:
                    action_idx = int(action_np)
                obs, reward, done, info = self.env.step(action_idx)
                step_count += 1
            
            sequences.append({
                'episode': episode,
                'latents': np.array(episode_latents),
                'factors': episode_factors,
                'steps': episode_steps
            })
        
        return sequences
    
    def simple_linear_regression_debug(self, X, y):
        """Simple linear regression with debug info."""
        # Add bias term
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Regularized normal equation
        reg_lambda = 1e-6
        XTX = X_with_bias.T @ X_with_bias
        XTX += reg_lambda * np.eye(XTX.shape[0])
        theta = np.linalg.solve(XTX, X_with_bias.T @ y)
        y_pred = X_with_bias @ theta
        
        # Calculate R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0
        mse = np.mean((y - y_pred) ** 2)
        
        return theta, y_pred, float(np.clip(r2, -1, 1)), float(max(0, mse))
    
    def debug_factor_over_time(self, sequences, factor_name, output_dir):
        """Debug a specific factor by plotting ground truth vs decoded over time."""
        print(f"Debugging factor: {factor_name}")
        
        # Collect all data
        all_latents = []
        all_factors = []
        sequence_info = []
        
        for seq in sequences:
            for i, (latent, factors) in enumerate(zip(seq['latents'], seq['factors'])):
                if factor_name in factors:
                    all_latents.append(latent)
                    all_factors.append(factors[factor_name])
                    sequence_info.append((seq['episode'], seq['steps'][i]))
        
        if len(all_latents) == 0:
            print(f"No data found for factor {factor_name}")
            return
        
        all_latents = np.array(all_latents)
        all_factors = np.array(all_factors)
        
        print(f"Factor {factor_name}:")
        print(f"  Data points: {len(all_factors)}")
        print(f"  Value range: [{np.min(all_factors):.4f}, {np.max(all_factors):.4f}]")
        print(f"  Variance: {np.var(all_factors):.6f}")
        print(f"  Unique values: {len(np.unique(all_factors))}")
        
        # Skip if no variation
        if np.var(all_factors) < 1e-6:
            print(f"  Skipping {factor_name} (no variation)")
            return
        
        # Train linear regression
        train_size = int(0.8 * len(all_latents))
        X_train, X_test = all_latents[:train_size], all_latents[train_size:]
        y_train, y_test = all_factors[:train_size], all_factors[train_size:]
        
        theta, y_pred_train, r2_train, mse_train = self.simple_linear_regression_debug(X_train, y_train)
        _, y_pred_test, r2_test, mse_test = self.simple_linear_regression_debug(X_test, y_test)
        
        print(f"  Train R²: {r2_train:.4f}, Test R²: {r2_test:.4f}")
        print(f"  Train MSE: {mse_train:.6f}, Test MSE: {mse_test:.6f}")
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Debug Analysis: {factor_name}', fontsize=16)
        
        # Plot 1: Ground truth over time
        ax1 = axes[0, 0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(sequences)))
        for i, seq in enumerate(sequences):
            factor_values = [f.get(factor_name, np.nan) for f in seq['factors']]
            ax1.plot(seq['steps'], factor_values, 'o-', color=colors[i], 
                    label=f'Episode {seq["episode"]}', alpha=0.7)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Ground Truth Value')
        ax1.set_title('Ground Truth Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Predicted vs Ground Truth (scatter)
        ax2 = axes[0, 1]
        ax2.scatter(y_train, y_pred_train, alpha=0.6, label=f'Train (R²={r2_train:.3f})')
        ax2.scatter(y_test, y_pred_test, alpha=0.6, label=f'Test (R²={r2_test:.3f})')
        min_val, max_val = min(np.min(all_factors), np.min(np.concatenate([y_pred_train, y_pred_test]))), \
                          max(np.max(all_factors), np.max(np.concatenate([y_pred_train, y_pred_test])))
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect prediction')
        ax2.set_xlabel('Ground Truth')
        ax2.set_ylabel('Predicted')
        ax2.set_title('Predicted vs Ground Truth')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Residuals over time
        ax3 = axes[1, 0]
        residuals_train = y_train - y_pred_train
        residuals_test = y_test - y_pred_test
        ax3.plot(range(len(residuals_train)), residuals_train, 'o', alpha=0.6, label='Train residuals')
        ax3.plot(range(len(residuals_train), len(residuals_train) + len(residuals_test)), 
                residuals_test, 'o', alpha=0.6, label='Test residuals')
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax3.set_xlabel('Data Point Index')
        ax3.set_ylabel('Residual (GT - Predicted)')
        ax3.set_title('Residuals Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Distribution of values
        ax4 = axes[1, 1]
        ax4.hist(all_factors, bins=20, alpha=0.7, label='Ground Truth', density=True)
        ax4.hist(np.concatenate([y_pred_train, y_pred_test]), bins=20, alpha=0.7, 
                label='Predicted', density=True)
        ax4.set_xlabel('Value')
        ax4.set_ylabel('Density')
        ax4.set_title('Value Distributions')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        plot_path = output_dir / f'debug_{factor_name.replace("/", "_")}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            'factor_name': factor_name,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mse_train': mse_train,
            'mse_test': mse_test,
            'data_points': len(all_factors),
            'variance': float(np.var(all_factors)),
            'unique_values': len(np.unique(all_factors)),
            'value_range': [float(np.min(all_factors)), float(np.max(all_factors))]
        }
    
    def debug_multiple_factors(self, sequences, output_dir, max_factors=10):
        """Debug multiple factors and create summary."""
        # Get all factor names
        all_factor_names = set()
        for seq in sequences:
            for factors in seq['factors']:
                all_factor_names.update(factors.keys())
        
        # Sort by variance to get most interesting factors
        factor_variances = {}
        for factor_name in all_factor_names:
            values = []
            for seq in sequences:
                for factors in seq['factors']:
                    if factor_name in factors:
                        values.append(factors[factor_name])
            if len(values) > 0:
                factor_variances[factor_name] = np.var(values)
        
        # Select top factors by variance
        sorted_factors = sorted(factor_variances.items(), key=lambda x: x[1], reverse=True)
        selected_factors = [name for name, var in sorted_factors[:max_factors] if var > 1e-6]
        
        print(f"Debugging top {len(selected_factors)} factors by variance:")
        for name, var in sorted_factors[:max_factors]:
            print(f"  {name}: variance = {var:.6f}")
        
        # Debug each factor
        results = []
        for factor_name in selected_factors:
            result = self.debug_factor_over_time(sequences, factor_name, output_dir)
            if result:
                results.append(result)
        
        # Create summary
        summary_path = output_dir / 'debug_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDebug analysis complete!")
        print(f"Results saved to: {output_dir}")
        print(f"Summary: {summary_path}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Debug linear probing analysis')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model directory')
    parser.add_argument('--config', type=str, default='vizdoom_deadly_corridor',
                       help='Configuration name')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to collect data from')
    parser.add_argument('--max-factors', type=int, default=10,
                       help='Maximum number of factors to debug')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = pathlib.Path(args.model_path) / 'debug_linear_probing'
    
    # Initialize debugger
    debugger = LinearProbingDebugger(args.model_path, args.config, args.device)
    
    # Collect sequence data
    sequences = debugger.collect_sequence_data(args.episodes)
    
    # Debug multiple factors
    results = debugger.debug_multiple_factors(sequences, args.output_dir, args.max_factors)


if __name__ == "__main__":
    main()
