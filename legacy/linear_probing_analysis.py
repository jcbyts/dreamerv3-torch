#!/usr/bin/env python3
"""
Linear probing analysis for VizDoom world model
Tests if ground truth factors can be linearly decoded from learned latent representations
"""

import argparse
import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from scipy.stats import pearsonr
import json

from enhanced_vizdoom import EnhancedViZDoom, extract_ground_truth_factors
from evaluate_agent import AgentEvaluator


class LinearProbingAnalyzer:
    def __init__(self, model_path, config_name, device='cuda'):
        self.model_path = pathlib.Path(model_path)
        self.device = device
        
        # Load the trained agent
        self.evaluator = AgentEvaluator(model_path, config_name, device)
        
        # Create enhanced environment with ground truth
        task = config_name.replace('vizdoom_', '')
        self.env = EnhancedViZDoom(task, enable_labels=True, enable_depth=True)
        
        print(f"Initialized linear probing for {task}")
        
    def collect_data(self, num_episodes=50, max_steps_per_episode=500):
        """Collect latent representations and ground truth factors."""
        print(f"Collecting data from {num_episodes} episodes...")
        
        latents = []
        factors = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            step_count = 0
            
            # Initialize agent state
            agent_state = None
            
            while not done and step_count < max_steps_per_episode:
                # Get latent representation from world model
                obs_tensor = {k: torch.tensor(v).unsqueeze(0).to(self.device) 
                             for k, v in obs.items() if k == 'image'}
                reset_tensor = torch.tensor([step_count == 0]).to(self.device)
                
                with torch.no_grad():
                    # Get latent state from world model
                    obs_processed = self.evaluator.agent._wm.preprocess(obs_tensor)
                    embed = self.evaluator.agent._wm.encoder(obs_processed)
                    
                    if agent_state is None:
                        latent = action = None
                    else:
                        latent, action = agent_state
                    
                    latent, _ = self.evaluator.agent._wm.dynamics.obs_step(
                        latent, action, embed, obs_tensor["is_first"] if "is_first" in obs_tensor else reset_tensor
                    )
                    
                    # Get feature representation
                    feat = self.evaluator.agent._wm.dynamics.get_feat(latent)
                    
                    # Get action for next step
                    policy_output, new_agent_state = self.evaluator.agent(
                        obs_tensor, reset_tensor, agent_state, training=False
                    )
                    action = policy_output["action"]
                    agent_state = new_agent_state
                
                # Extract latent vector (concatenate stochastic and deterministic parts)
                latent_vector = feat.cpu().numpy().flatten()
                latents.append(latent_vector)
                
                # Extract ground truth factors
                gt_factors = extract_ground_truth_factors(obs)
                factors.append(gt_factors)
                
                # Step environment
                obs, reward, done, info = self.env.step(action.cpu().numpy()[0])
                step_count += 1
            
            if episode % 10 == 0:
                print(f"Completed episode {episode}/{num_episodes}")
        
        print(f"Collected {len(latents)} data points")
        return np.array(latents), factors
    
    def analyze_linear_decodability(self, latents, factors, test_size=0.2):
        """Analyze which ground truth factors can be linearly decoded from latents."""
        print("Analyzing linear decodability...")
        
        # Convert factors to arrays
        factor_names = list(factors[0].keys())
        factor_arrays = {}
        
        for name in factor_names:
            values = [f[name] for f in factors]
            factor_arrays[name] = np.array(values)
        
        results = {}
        
        for factor_name, factor_values in factor_arrays.items():
            print(f"\nAnalyzing factor: {factor_name}")
            
            # Skip if factor has no variation
            if np.var(factor_values) < 1e-6:
                print(f"  Skipping {factor_name} (no variation)")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                latents, factor_values, test_size=test_size, random_state=42
            )
            
            # Determine if classification or regression
            unique_values = np.unique(factor_values)
            is_binary = len(unique_values) == 2 and set(unique_values) <= {0, 1}
            is_categorical = len(unique_values) < 10 and len(unique_values) < len(factor_values) * 0.1
            
            if is_binary or is_categorical:
                # Classification
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                results[factor_name] = {
                    'type': 'classification',
                    'accuracy': accuracy,
                    'baseline_accuracy': max(np.bincount(y_train)) / len(y_train),
                    'improvement': accuracy - max(np.bincount(y_train)) / len(y_train),
                    'unique_values': unique_values.tolist()
                }
                
                print(f"  Classification accuracy: {accuracy:.3f}")
                print(f"  Baseline accuracy: {results[factor_name]['baseline_accuracy']:.3f}")
                print(f"  Improvement: {results[factor_name]['improvement']:.3f}")
                
            else:
                # Regression
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                r2 = r2_score(y_test, y_pred)
                mse = np.mean((y_test - y_pred) ** 2)
                
                results[factor_name] = {
                    'type': 'regression',
                    'r2_score': r2,
                    'mse': mse,
                    'mean_value': np.mean(factor_values),
                    'std_value': np.std(factor_values)
                }
                
                print(f"  R² score: {r2:.3f}")
                print(f"  MSE: {mse:.6f}")
        
    def create_comprehensive_analysis(self, latents, factors, results, output_dir):
        """Create comprehensive analysis with all visualizations."""
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        print("Creating comprehensive analysis...")

        # Save results to JSON
        results_path = output_dir / 'linear_probing_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # 1. Mutual Information Heatmap
        mi_matrix, factor_names = self.compute_mutual_information(latents, factors)
        mi_plot_path = self.create_mutual_information_plot(mi_matrix, factor_names, output_dir)

        # 2. Linear Decodability Bar Chart
        decodability_plot_path = self.create_linear_decodability_plot(results, output_dir)

        # 3. Disentanglement Metrics
        disentanglement_metrics = self.compute_disentanglement_metrics(latents, factors)
        disentanglement_plot_path = self.create_disentanglement_plot(disentanglement_metrics, output_dir)

        # 4. Create detailed report
        report_path = output_dir / 'comprehensive_analysis_report.txt'
        with open(report_path, 'w') as f:
            f.write("Comprehensive Linear Probing Analysis Report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Model: {self.model_path}\n")
            f.write(f"Total factors analyzed: {len(results)}\n")
            f.write(f"Latent dimensions: {latents.shape[1]}\n")
            f.write(f"Data points collected: {latents.shape[0]}\n\n")

            # Disentanglement metrics summary
            if disentanglement_metrics:
                f.write("Disentanglement Metrics:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Informativeness: {disentanglement_metrics['informativeness']:.3f}\n")
                f.write(f"Disentanglement: {disentanglement_metrics['disentanglement']:.3f}\n")
                f.write(f"Completeness: {disentanglement_metrics['completeness']:.3f}\n\n")

            # Linear decodability summary
            classification_results = [v for v in results.values() if v['type'] == 'classification']
            regression_results = [v for v in results.values() if v['type'] == 'regression']

            if classification_results:
                avg_accuracy = np.mean([r['accuracy'] for r in classification_results])
                avg_improvement = np.mean([r['improvement'] for r in classification_results])
                f.write(f"Classification factors: {len(classification_results)}\n")
                f.write(f"Average accuracy: {avg_accuracy:.3f}\n")
                f.write(f"Average improvement: {avg_improvement:.3f}\n\n")

            if regression_results:
                avg_r2 = np.mean([r['r2_score'] for r in regression_results])
                f.write(f"Regression factors: {len(regression_results)}\n")
                f.write(f"Average R² score: {avg_r2:.3f}\n\n")

            # Mutual information summary
            f.write("Mutual Information Analysis:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average MI across all factor-latent pairs: {np.mean(mi_matrix):.4f}\n")
            f.write(f"Maximum MI: {np.max(mi_matrix):.4f}\n")
            f.write(f"Factors with highest average MI:\n")

            avg_mi_per_factor = np.mean(mi_matrix, axis=1)
            sorted_indices = np.argsort(avg_mi_per_factor)[::-1]
            for i in sorted_indices[:5]:
                f.write(f"  {factor_names[i]}: {avg_mi_per_factor[i]:.4f}\n")

            # Detailed results
            f.write("\nDetailed Linear Decodability Results:\n")
            f.write("-" * 40 + "\n")

            for factor_name, result in results.items():
                f.write(f"\n{factor_name}:\n")
                if result['type'] == 'classification':
                    f.write(f"  Type: Classification\n")
                    f.write(f"  Accuracy: {result['accuracy']:.3f}\n")
                    f.write(f"  Baseline: {result['baseline_accuracy']:.3f}\n")
                    f.write(f"  Improvement: {result['improvement']:.3f}\n")
                else:
                    f.write(f"  Type: Regression\n")
                    f.write(f"  R² score: {result['r2_score']:.3f}\n")
                    f.write(f"  MSE: {result['mse']:.6f}\n")

        print(f"\nComprehensive analysis complete!")
        print(f"Results saved to: {output_dir}")
        print(f"Mutual information heatmap: {mi_plot_path}")
        print(f"Linear decodability plot: {decodability_plot_path}")
        if disentanglement_plot_path:
            print(f"Disentanglement metrics plot: {disentanglement_plot_path}")
        print(f"Detailed report: {report_path}")

        return {
            'results': results,
            'mi_matrix': mi_matrix,
            'factor_names': factor_names,
            'disentanglement_metrics': disentanglement_metrics
        }
    
    def compute_mutual_information(self, latents, factors):
        """Compute mutual information between latent dimensions and ground truth factors."""
        print("Computing mutual information...")

        # Convert factors to arrays
        factor_names = list(factors[0].keys())
        factor_arrays = {}

        for name in factor_names:
            values = [f[name] for f in factors]
            factor_arrays[name] = np.array(values)

        # Compute MI for each latent dimension vs each factor
        n_latents = latents.shape[1]
        mi_matrix = np.zeros((len(factor_names), n_latents))

        for i, (factor_name, factor_values) in enumerate(factor_arrays.items()):
            # Skip if factor has no variation
            if np.var(factor_values) < 1e-6:
                continue

            # Determine if classification or regression
            unique_values = np.unique(factor_values)
            is_categorical = len(unique_values) < 10 and len(unique_values) < len(factor_values) * 0.1

            for j in range(n_latents):
                latent_dim = latents[:, j]

                if is_categorical:
                    # Use mutual info for classification
                    mi = mutual_info_classif(latent_dim.reshape(-1, 1), factor_values, random_state=42)[0]
                else:
                    # Use mutual info for regression
                    mi = mutual_info_regression(latent_dim.reshape(-1, 1), factor_values, random_state=42)[0]

                mi_matrix[i, j] = mi

        return mi_matrix, factor_names

    def create_mutual_information_plot(self, mi_matrix, factor_names, output_dir):
        """Create mutual information heatmap like Figure 3."""
        plt.figure(figsize=(15, 8))

        # Create heatmap
        sns.heatmap(mi_matrix,
                   xticklabels=[f'L{i}' for i in range(mi_matrix.shape[1])],
                   yticklabels=factor_names,
                   cmap='Blues',
                   cbar_kws={'label': 'Mutual Information'})

        plt.title('Mutual Information between Latent Variables and Ground Truth Factors')
        plt.xlabel('Latent Dimensions')
        plt.ylabel('Ground Truth Factors')
        plt.tight_layout()

        plot_path = output_dir / 'mutual_information_heatmap.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return plot_path

    def create_linear_decodability_plot(self, results, output_dir):
        """Create linear decodability bar chart like Figure 4."""
        # Combine all results into R² scores
        factor_names = []
        r2_scores = []

        for factor_name, result in results.items():
            factor_names.append(factor_name)
            if result['type'] == 'classification':
                # Convert accuracy to R²-like score
                r2_scores.append(result['accuracy'])
            else:
                r2_scores.append(max(0, result['r2_score']))  # Clip negative R²

        # Create bar plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(factor_names)), r2_scores,
                      color='steelblue', alpha=0.7)

        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, r2_scores)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=10)

        plt.xlabel('Ground Truth Factors')
        plt.ylabel('Linear Decodability (R² / Accuracy)')
        plt.title('Linear Decodability of Ground Truth Factors from Latent Representations')
        plt.xticks(range(len(factor_names)), factor_names, rotation=45, ha='right')
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        plot_path = output_dir / 'linear_decodability.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return plot_path

    def compute_disentanglement_metrics(self, latents, factors):
        """Compute disentanglement metrics like informativeness, disentanglement, completeness."""
        print("Computing disentanglement metrics...")

        # Convert factors to arrays
        factor_names = list(factors[0].keys())
        factor_arrays = {}

        for name in factor_names:
            values = [f[name] for f in factors]
            factor_arrays[name] = np.array(values)

        # Remove factors with no variation
        factor_arrays = {k: v for k, v in factor_arrays.items() if np.var(v) > 1e-6}
        factor_names = list(factor_arrays.keys())

        if len(factor_names) == 0:
            return None

        # Compute importance matrix (R² scores)
        importance_matrix = np.zeros((len(factor_names), latents.shape[1]))

        for i, (factor_name, factor_values) in enumerate(factor_arrays.items()):
            for j in range(latents.shape[1]):
                latent_dim = latents[:, j].reshape(-1, 1)

                # Use linear regression to compute R²
                reg = LinearRegression()
                reg.fit(latent_dim, factor_values)
                r2 = reg.score(latent_dim, factor_values)
                importance_matrix[i, j] = max(0, r2)

        # Compute metrics
        # Informativeness: average R² across all factor-latent pairs
        informativeness = np.mean(importance_matrix)

        # Disentanglement: how much each latent focuses on one factor
        disentanglement_scores = []
        for j in range(importance_matrix.shape[1]):
            latent_scores = importance_matrix[:, j]
            if np.sum(latent_scores) > 0:
                # Entropy-based disentanglement
                normalized_scores = latent_scores / np.sum(latent_scores)
                entropy = -np.sum(normalized_scores * np.log(normalized_scores + 1e-12))
                max_entropy = np.log(len(factor_names))
                disentanglement = 1 - entropy / max_entropy if max_entropy > 0 else 0
                disentanglement_scores.append(disentanglement)

        disentanglement = np.mean(disentanglement_scores) if disentanglement_scores else 0

        # Completeness: how much each factor is captured by latents
        completeness_scores = []
        for i in range(importance_matrix.shape[0]):
            factor_scores = importance_matrix[i, :]
            if np.sum(factor_scores) > 0:
                # Entropy-based completeness
                normalized_scores = factor_scores / np.sum(factor_scores)
                entropy = -np.sum(normalized_scores * np.log(normalized_scores + 1e-12))
                max_entropy = np.log(importance_matrix.shape[1])
                completeness = 1 - entropy / max_entropy if max_entropy > 0 else 0
                completeness_scores.append(completeness)

        completeness = np.mean(completeness_scores) if completeness_scores else 0

        return {
            'informativeness': informativeness,
            'disentanglement': disentanglement,
            'completeness': completeness,
            'importance_matrix': importance_matrix,
            'factor_names': factor_names
        }

    def create_disentanglement_plot(self, disentanglement_metrics, output_dir):
        """Create disentanglement metrics plot like Figure 5."""
        if disentanglement_metrics is None:
            return None

        metrics = ['Informativeness', 'Disentanglement', 'Completeness']
        values = [
            disentanglement_metrics['informativeness'],
            disentanglement_metrics['disentanglement'],
            disentanglement_metrics['completeness']
        ]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, color=['steelblue', 'orange', 'green'], alpha=0.7)

        # Add value labels
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.ylabel('Score')
        plt.title('Disentanglement Metrics for Learned Representations')
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        plot_path = output_dir / 'disentanglement_metrics.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return plot_path


def main():
    parser = argparse.ArgumentParser(description='Linear probing analysis for VizDoom world model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model directory')
    parser.add_argument('--config', type=str, default='vizdoom_basic',
                       help='Configuration name')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of episodes to collect data from')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = pathlib.Path(args.model_path) / 'linear_probing_analysis'
    
    # Initialize analyzer
    analyzer = LinearProbingAnalyzer(args.model_path, args.config, args.device)
    
    # Collect data
    latents, factors = analyzer.collect_data(args.episodes)

    # Analyze linear decodability
    results = analyzer.analyze_linear_decodability(latents, factors)

    # Create comprehensive analysis with all visualizations
    analyzer.create_comprehensive_analysis(latents, factors, results, args.output_dir)


if __name__ == "__main__":
    main()
