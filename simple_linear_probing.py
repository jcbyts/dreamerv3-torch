#!/usr/bin/env python3
"""
Simple linear probing analysis for VizDoom world model without sklearn dependencies
Tests if ground truth factors can be linearly decoded from learned latent representations
"""

import argparse
import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import json

from enhanced_vizdoom import EnhancedViZDoom, extract_minimal_vizdoom_factors
from evaluate_agent import AgentEvaluator


class SimpleLinearProbingAnalyzer:
    def __init__(self, model_path, config_name, device='cuda'):
        self.model_path = pathlib.Path(model_path)
        self.device = device
        
        # Load the trained agent
        self.evaluator = AgentEvaluator(model_path, config_name, device)
        
        # Create enhanced environment with ground truth
        task = config_name.replace('vizdoom_', '')
        self.env = EnhancedViZDoom(task, enable_labels=True, enable_depth=True)
        
        print(f"Initialized linear probing for {task}")
        
    def collect_data(self, num_episodes=50, max_steps_per_episode=200):
        """Collect latent representations and ground truth factors."""
        print(f"Collecting data from {num_episodes} episodes...")
        
        latents = []
        factors = []
        episode_ids = []  # Track which episode each sample comes from

        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            step_count = 0

            # Initialize agent state and previous observation tracking
            agent_state = None
            self._prev_obs_vars = None
            
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
                
                # Extract latent vector (concatenate stochastic and deterministic parts)
                latent_vector = feat.cpu().numpy().flatten()
                latents.append(latent_vector)
                episode_ids.append(episode)  # Track episode ID for each sample

                # Debug: Print dimensions on first step
                if len(latents) == 1:
                    print(f"Latent vector dimension: {len(latent_vector)}")
                    print(f"First few latent values: {latent_vector[:10]}")

                # Extract ground truth factors with previous state for deltas
                gt_factors = extract_minimal_vizdoom_factors(obs, self._prev_obs_vars)
                factors.append(gt_factors)

                # Update previous state for next iteration
                if 'game_variables' in obs:
                    self._prev_obs_vars = obs['game_variables'].copy()
                else:
                    self._prev_obs_vars = None
                
                # Step environment
                action_np = action.cpu().numpy()
                if action_np.ndim > 0:
                    action_idx = int(action_np.flatten()[0])
                else:
                    action_idx = int(action_np)
                obs, _, done, _ = self.env.step(action_idx)
                step_count += 1
            
            if episode % 5 == 0:
                print(f"Completed episode {episode}/{num_episodes}")
        
        print(f"Collected {len(latents)} data points")
        latents_array = np.array(latents)
        episode_ids_array = np.array(episode_ids)
        print(f"Latent array shape: {latents_array.shape}")
        print(f"Data points: {latents_array.shape[0]}, Latent dimensions: {latents_array.shape[1]}")
        print(f"Episodes: {len(np.unique(episode_ids_array))}")

        # Return raw data - PCA will be applied after train/test split to avoid leakage
        return latents_array, factors, episode_ids_array
    
    def simple_linear_regression(self, X, y):
        """Simple linear regression using normal equations."""
        # Add bias term
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])

        # Add regularization to avoid numerical issues
        reg_lambda = 1e-6

        try:
            # Regularized normal equation: theta = (X^T X + λI)^(-1) X^T y
            XTX = X_with_bias.T @ X_with_bias
            XTX += reg_lambda * np.eye(XTX.shape[0])
            theta = np.linalg.solve(XTX, X_with_bias.T @ y)
            y_pred = X_with_bias @ theta

            # Calculate R²
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0
            mse = np.mean((y - y_pred) ** 2)

            # Ensure finite values
            r2 = float(np.clip(r2, -1, 1))
            mse = float(max(0, mse))

            return r2, mse
        except (np.linalg.LinAlgError, ValueError):
            return 0.0, float(np.var(y))
    
    def analyze_linear_decodability(self, latents, factors, episode_ids, test_ratio=0.2, run_sanity_checks=True):
        """Analyze which ground truth factors can be linearly decoded from latents."""
        print("Analyzing linear decodability...")

        # Episode-based train/test split to avoid temporal leakage
        unique_episodes = np.unique(episode_ids)
        np.random.shuffle(unique_episodes)
        n_test_episodes = max(1, int(len(unique_episodes) * test_ratio))
        test_episodes = set(unique_episodes[:n_test_episodes])

        train_mask = ~np.isin(episode_ids, list(test_episodes))
        test_mask = np.isin(episode_ids, list(test_episodes))

        print(f"Episode-based split: {len(unique_episodes)} total episodes")
        print(f"Train episodes: {np.sum(train_mask)} samples from {len(unique_episodes) - n_test_episodes} episodes")
        print(f"Test episodes: {np.sum(test_mask)} samples from {n_test_episodes} episodes")

        # Check if we need dimensionality reduction
        n_train_samples = np.sum(train_mask)
        n_latent_dims = latents.shape[1]

        print(f"Training samples: {n_train_samples}, Latent dimensions: {n_latent_dims}")
        print(f"Samples-to-dimensions ratio: {n_train_samples / n_latent_dims:.2f}")

        # Only apply PCA if we have fewer than 3x samples per dimension
        if n_latent_dims > n_train_samples // 3:
            print("⚠️  WARNING: Low samples-to-dimensions ratio - applying PCA")
            n_components = min(n_train_samples // 5, 200)  # Use 1/5 of samples, max 200

            # Simple PCA implementation
            X_train = latents[train_mask]
            X_test = latents[test_mask]

            # Center the training data
            train_mean = np.mean(X_train, axis=0)
            X_train_centered = X_train - train_mean
            X_test_centered = X_test - train_mean

            # Compute covariance matrix and eigendecomposition
            cov_matrix = np.cov(X_train_centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # Sort by eigenvalues (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Select top components
            components = eigenvectors[:, :n_components]

            # Transform data
            latents_train_pca = X_train_centered @ components
            latents_test_pca = X_test_centered @ components

            explained_variance_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)

            print(f"PCA: {n_latent_dims} → {n_components} dimensions")
            print(f"Explained variance: {np.sum(explained_variance_ratio):.3f}")

            latents_processed = np.zeros((len(latents), n_components))
            latents_processed[train_mask] = latents_train_pca
            latents_processed[test_mask] = latents_test_pca
        else:
            print("✅ Good samples-to-dimensions ratio - using full latent space")
            latents_processed = latents

        # Normalize latent vectors (fit normalization on training data only)
        train_mean = np.mean(latents_processed[train_mask], axis=0)
        train_std = np.std(latents_processed[train_mask], axis=0) + 1e-8
        latents_normalized = (latents_processed - train_mean) / train_std

        print(f"Normalization: train mean={np.mean(train_mean):.6f}, train std={np.mean(train_std):.6f}")

        # Convert factors to arrays, handling missing keys
        all_factor_names = set()
        for f in factors:
            all_factor_names.update(f.keys())

        factor_arrays = {}
        for name in all_factor_names:
            values = [f.get(name, 0.0) for f in factors]  # Default to 0.0 if missing
            factor_arrays[name] = np.array(values)

        results = {}
        sanity_results = {}

        X_train, X_test = latents_normalized[train_mask], latents_normalized[test_mask]
        
        for factor_name, factor_values in factor_arrays.items():
            print(f"\nAnalyzing factor: {factor_name}")
            
            # Skip if factor has no variation
            if np.var(factor_values) < 1e-6:
                print(f"  Skipping {factor_name} (no variation)")
                continue
            
            y_train, y_test = factor_values[train_mask], factor_values[test_mask]
            
            # Fit linear regression on train data only
            # Add bias term to training data
            X_train_with_bias = np.column_stack([np.ones(X_train.shape[0]), X_train])

            # Regularized normal equation: theta = (X^T X + λI)^(-1) X^T y
            reg_lambda = 1e-6
            XTX = X_train_with_bias.T @ X_train_with_bias
            XTX += reg_lambda * np.eye(XTX.shape[0])
            theta = np.linalg.solve(XTX, X_train_with_bias.T @ y_train)

            # Evaluate on training data
            y_pred_train = X_train_with_bias @ theta
            ss_res_train = np.sum((y_train - y_pred_train) ** 2)
            ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
            r2_train = 1 - (ss_res_train / ss_tot_train) if ss_tot_train > 1e-10 else 0
            mse_train = np.mean((y_train - y_pred_train) ** 2)

            # Evaluate on test data (using model trained on train data)
            X_test_with_bias = np.column_stack([np.ones(X_test.shape[0]), X_test])
            y_pred_test = X_test_with_bias @ theta
            ss_res_test = np.sum((y_test - y_pred_test) ** 2)
            ss_tot_test = np.sum((y_test - np.mean(y_test)) ** 2)
            r2_test = 1 - (ss_res_test / ss_tot_test) if ss_tot_test > 1e-10 else 0
            mse_test = np.mean((y_test - y_pred_test) ** 2)

            # Ensure finite values (don't clip R² - let us see if there are issues)
            r2_train = float(r2_train) if np.isfinite(r2_train) else -999.0
            r2_test = float(r2_test) if np.isfinite(r2_test) else -999.0
            mse_train = float(mse_train) if np.isfinite(mse_train) else float('inf')
            mse_test = float(mse_test) if np.isfinite(mse_test) else float('inf')
            
            results[factor_name] = {
                'r2_train': float(r2_train),
                'r2_test': float(r2_test),
                'mse_train': float(mse_train),
                'mse_test': float(mse_test),
                'mean_value': float(np.mean(factor_values)),
                'std_value': float(np.std(factor_values))
            }
            
            print(f"  Train R²: {r2_train:.3f}, Test R²: {r2_test:.3f}")
            print(f"  Train MSE: {mse_train:.6f}, Test MSE: {mse_test:.6f}")

            # Sanity check: test with random permutation of factor values
            if run_sanity_checks and r2_test > 0.8:  # Only check suspiciously high scores
                y_train_shuffled = np.random.permutation(y_train)
                y_test_shuffled = np.random.permutation(y_test)

                # Fit on shuffled train data
                theta_shuffled = np.linalg.solve(XTX, X_train_with_bias.T @ y_train_shuffled)
                y_pred_test_shuffled = X_test_with_bias @ theta_shuffled
                ss_res_shuffled = np.sum((y_test_shuffled - y_pred_test_shuffled) ** 2)
                ss_tot_shuffled = np.sum((y_test_shuffled - np.mean(y_test_shuffled)) ** 2)
                r2_shuffled = 1 - (ss_res_shuffled / ss_tot_shuffled) if ss_tot_shuffled > 1e-10 else 0

                sanity_results[factor_name] = {
                    'r2_shuffled': float(r2_shuffled),
                    'original_r2': float(r2_test)
                }

                if r2_shuffled > 0.5:
                    print(f"  🚨 SANITY CHECK FAILED: Shuffled R² = {r2_shuffled:.3f} (should be ~0)")
                else:
                    print(f"  ✅ Sanity check passed: Shuffled R² = {r2_shuffled:.3f}")

        # Print sanity check summary
        if run_sanity_checks and sanity_results:
            print(f"\nSanity Check Summary:")
            print(f"Factors tested: {len(sanity_results)}")
            failed_checks = [name for name, result in sanity_results.items() if result['r2_shuffled'] > 0.5]
            if failed_checks:
                print(f"🚨 FAILED sanity checks: {failed_checks}")
            else:
                print("✅ All sanity checks passed")

        return results
    
    def create_visualization(self, results, output_dir):
        """Create visualization of linear decodability results."""
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save results to JSON
        results_path = output_dir / 'simple_linear_probing_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create bar plot
        factor_names = list(results.keys())
        test_r2_scores = [results[name]['r2_test'] for name in factor_names]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(factor_names)), test_r2_scores, 
                      color='steelblue', alpha=0.7)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, test_r2_scores)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Ground Truth Factors')
        plt.ylabel('Test R² Score')
        plt.title('Linear Decodability of Ground Truth Factors from Latent Representations')
        plt.xticks(range(len(factor_names)), factor_names, rotation=45, ha='right')
        plt.ylim(0, max(1.0, max(test_r2_scores) * 1.1))
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plot_path = output_dir / 'simple_linear_decodability.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create summary report
        report_path = output_dir / 'simple_analysis_report.txt'
        with open(report_path, 'w') as f:
            f.write("Simple Linear Probing Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Total factors analyzed: {len(results)}\n\n")
            
            # Summary statistics
            avg_test_r2 = np.mean([r['r2_test'] for r in results.values()])
            f.write(f"Average test R² score: {avg_test_r2:.3f}\n\n")
            
            # Top performing factors
            sorted_factors = sorted(results.items(), key=lambda x: x[1]['r2_test'], reverse=True)
            f.write("Top 5 best decoded factors:\n")
            for i, (factor_name, result) in enumerate(sorted_factors[:5]):
                f.write(f"  {i+1}. {factor_name}: R² = {result['r2_test']:.3f}\n")
            
            f.write("\nDetailed Results:\n")
            f.write("-" * 30 + "\n")
            
            for factor_name, result in results.items():
                f.write(f"\n{factor_name}:\n")
                f.write(f"  Train R²: {result['r2_train']:.3f}\n")
                f.write(f"  Test R²: {result['r2_test']:.3f}\n")
                f.write(f"  Train MSE: {result['mse_train']:.6f}\n")
                f.write(f"  Test MSE: {result['mse_test']:.6f}\n")
                f.write(f"  Mean value: {result['mean_value']:.4f}\n")
                f.write(f"  Std value: {result['std_value']:.4f}\n")
        
        print(f"\nAnalysis complete!")
        print(f"Results saved to: {output_dir}")
        print(f"Plot: {plot_path}")
        print(f"Report: {report_path}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Simple linear probing analysis for VizDoom world model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model directory')
    parser.add_argument('--config', type=str, default='vizdoom_basic',
                       help='Configuration name')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Number of episodes to collect data from')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = pathlib.Path(args.model_path) / 'simple_linear_probing_analysis'
    
    # Initialize analyzer
    analyzer = SimpleLinearProbingAnalyzer(args.model_path, args.config, args.device)
    
    # Collect data
    latents, factors, episode_ids = analyzer.collect_data(args.episodes)

    # Analyze linear decodability
    results = analyzer.analyze_linear_decodability(latents, factors, episode_ids)

    # Create visualization
    analyzer.create_visualization(results, args.output_dir)


if __name__ == "__main__":
    main()
