"""
Task Performance Evaluation Module

Evaluates models on task-specific metrics:
- Episode returns (mean, std, median, max)
- Success rates (task completion)
- Convergence speed (steps to threshold)
- Learning stability (variance over time)
- Final performance (last N episodes)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TaskPerformanceEvaluator:
    """Evaluates task performance metrics for model comparison."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.num_episodes = config.get('num_episodes', 100)
        self.num_seeds = config.get('num_seeds', 5)
        self.confidence_level = config.get('confidence_level', 0.95)
        
    def evaluate_model(self, model, environment, num_episodes: Optional[int] = None) -> Dict:
        """
        Evaluate a single model on task performance.
        
        Args:
            model: Trained model to evaluate
            environment: Environment to evaluate on
            num_episodes: Number of episodes to run (overrides config)
            
        Returns:
            Dict containing performance metrics
        """
        num_episodes = num_episodes or self.num_episodes
        
        # Run evaluation episodes
        episode_returns = []
        episode_lengths = []
        success_flags = []
        
        model.eval()
        with torch.no_grad():
            for episode in range(num_episodes):
                obs = environment.reset()
                episode_return = 0.0
                episode_length = 0
                done = False
                
                while not done:
                    # Get action from model
                    action = self._get_model_action(model, obs)
                    obs, reward, done, info = environment.step(action)
                    
                    episode_return += reward
                    episode_length += 1
                    
                    # Check for episode timeout
                    if episode_length >= environment.max_episode_steps:
                        break
                
                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)
                
                # Determine success (task-specific)
                success = self._determine_success(episode_return, info, environment)
                success_flags.append(success)
                
                if episode % 20 == 0:
                    logger.info(f"Episode {episode}/{num_episodes}, Return: {episode_return:.2f}")
        
        # Calculate metrics
        metrics = self._calculate_performance_metrics(
            episode_returns, episode_lengths, success_flags
        )
        
        return metrics
    
    def compare_models(self, model1, model2, environments: List, 
                      model1_name: str = "DreamerV3", model2_name: str = "hDreamer") -> Dict:
        """
        Compare two models across multiple environments.
        
        Args:
            model1: First model (typically DreamerV3)
            model2: Second model (typically hDreamer)
            environments: List of environments to evaluate on
            model1_name: Name for first model
            model2_name: Name for second model
            
        Returns:
            Dict containing comparison results
        """
        comparison_results = {}
        
        for env in environments:
            env_name = env.__class__.__name__
            logger.info(f"Comparing models on {env_name}")
            
            # Evaluate both models
            model1_results = self.evaluate_model(model1, env)
            model2_results = self.evaluate_model(model2, env)
            
            # Statistical comparison
            comparison = self._statistical_comparison(
                model1_results, model2_results, model1_name, model2_name
            )
            
            comparison_results[env_name] = {
                model1_name.lower(): model1_results,
                model2_name.lower(): model2_results,
                'comparison': comparison
            }
        
        # Aggregate results across environments
        comparison_results['aggregate'] = self._aggregate_comparison(
            comparison_results, model1_name, model2_name
        )
        
        return comparison_results
    
    def _get_model_action(self, model, obs):
        """Get action from model given observation."""
        # Convert observation to model format
        if isinstance(obs, dict):
            # Handle dict observations (e.g., VizDoom)
            obs_tensor = {}
            for key, value in obs.items():
                if key == 'image':
                    # Normalize image and add batch dimension
                    obs_tensor[key] = torch.tensor(value, dtype=torch.float32).unsqueeze(0) / 255.0
                else:
                    obs_tensor[key] = torch.tensor(value, dtype=torch.float32).unsqueeze(0)
        else:
            # Handle array observations
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        # Get action from model
        with torch.no_grad():
            action_dist = model.actor(obs_tensor)
            action = action_dist.mode()  # Use deterministic action for evaluation
        
        return action.squeeze(0).cpu().numpy()
    
    def _determine_success(self, episode_return: float, info: Dict, environment) -> bool:
        """Determine if episode was successful (task-specific)."""
        # Default: success if return above threshold
        success_threshold = getattr(environment, 'success_threshold', 0.0)
        
        # Check info dict for explicit success flag
        if 'success' in info:
            return info['success']
        elif 'is_success' in info:
            return info['is_success']
        else:
            # Use return threshold
            return episode_return >= success_threshold
    
    def _calculate_performance_metrics(self, returns: List[float], 
                                     lengths: List[int], 
                                     successes: List[bool]) -> Dict:
        """Calculate comprehensive performance metrics."""
        returns = np.array(returns)
        lengths = np.array(lengths)
        successes = np.array(successes)
        
        metrics = {
            # Return statistics
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'median_return': float(np.median(returns)),
            'max_return': float(np.max(returns)),
            'min_return': float(np.min(returns)),
            'q75_return': float(np.percentile(returns, 75)),
            'q25_return': float(np.percentile(returns, 25)),
            
            # Episode length statistics
            'mean_length': float(np.mean(lengths)),
            'std_length': float(np.std(lengths)),
            
            # Success metrics
            'success_rate': float(np.mean(successes)),
            'num_successes': int(np.sum(successes)),
            'total_episodes': len(returns),
            
            # Stability metrics
            'return_cv': float(np.std(returns) / np.mean(returns)) if np.mean(returns) != 0 else float('inf'),
            'return_iqr': float(np.percentile(returns, 75) - np.percentile(returns, 25)),
            
            # Raw data for further analysis
            'all_returns': returns.tolist(),
            'all_lengths': lengths.tolist(),
            'all_successes': successes.tolist()
        }
        
        return metrics
    
    def _statistical_comparison(self, results1: Dict, results2: Dict, 
                               name1: str, name2: str) -> Dict:
        """Perform statistical comparison between two sets of results."""
        returns1 = np.array(results1['all_returns'])
        returns2 = np.array(results2['all_returns'])
        
        # T-test for mean difference
        t_stat, t_pvalue = stats.ttest_ind(returns1, returns2)
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_pvalue = stats.mannwhitneyu(returns1, returns2, alternative='two-sided')
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(returns1) - 1) * np.var(returns1, ddof=1) + 
                             (len(returns2) - 1) * np.var(returns2, ddof=1)) / 
                            (len(returns1) + len(returns2) - 2))
        cohens_d = (np.mean(returns1) - np.mean(returns2)) / pooled_std if pooled_std != 0 else 0
        
        # Determine winner
        if results1['mean_return'] > results2['mean_return']:
            winner = name1
            improvement = ((results1['mean_return'] - results2['mean_return']) / 
                          results2['mean_return'] * 100)
        elif results2['mean_return'] > results1['mean_return']:
            winner = name2
            improvement = ((results2['mean_return'] - results1['mean_return']) / 
                          results1['mean_return'] * 100)
        else:
            winner = "tie"
            improvement = 0.0
        
        comparison = {
            'winner': winner,
            'improvement_percent': float(improvement),
            'statistical_tests': {
                't_test': {'statistic': float(t_stat), 'p_value': float(t_pvalue)},
                'mann_whitney': {'statistic': float(u_stat), 'p_value': float(u_pvalue)},
                'cohens_d': float(cohens_d)
            },
            'significant': float(t_pvalue) < (1 - self.confidence_level),
            'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
        }
        
        return comparison
    
    def _aggregate_comparison(self, env_results: Dict, name1: str, name2: str) -> Dict:
        """Aggregate comparison results across environments."""
        wins_model1 = 0
        wins_model2 = 0
        ties = 0
        
        significant_wins_model1 = 0
        significant_wins_model2 = 0
        
        for env_name, results in env_results.items():
            if env_name == 'aggregate':
                continue
                
            comparison = results['comparison']
            winner = comparison['winner']
            significant = comparison['significant']
            
            if winner == name1:
                wins_model1 += 1
                if significant:
                    significant_wins_model1 += 1
            elif winner == name2:
                wins_model2 += 1
                if significant:
                    significant_wins_model2 += 1
            else:
                ties += 1
        
        total_envs = wins_model1 + wins_model2 + ties
        
        aggregate = {
            'total_environments': total_envs,
            'wins': {
                name1: wins_model1,
                name2: wins_model2,
                'ties': ties
            },
            'significant_wins': {
                name1: significant_wins_model1,
                name2: significant_wins_model2
            },
            'win_rate': {
                name1: wins_model1 / total_envs if total_envs > 0 else 0,
                name2: wins_model2 / total_envs if total_envs > 0 else 0
            },
            'overall_winner': name1 if wins_model1 > wins_model2 else name2 if wins_model2 > wins_model1 else 'tie'
        }
        
        return aggregate
    
    def plot_comparison(self, comparison_results: Dict, output_path: str):
        """Generate comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data for plotting
        env_names = [name for name in comparison_results.keys() if name != 'aggregate']
        model1_returns = []
        model2_returns = []
        
        for env_name in env_names:
            results = comparison_results[env_name]
            model1_returns.append(results['dreamerv3']['mean_return'])
            model2_returns.append(results['hdreamer']['mean_return'])
        
        # Plot 1: Mean returns comparison
        x = np.arange(len(env_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, model1_returns, width, label='DreamerV3', alpha=0.8)
        axes[0, 0].bar(x + width/2, model2_returns, width, label='hDreamer', alpha=0.8)
        axes[0, 0].set_xlabel('Environment')
        axes[0, 0].set_ylabel('Mean Return')
        axes[0, 0].set_title('Mean Return Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(env_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add more plots as needed...
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
