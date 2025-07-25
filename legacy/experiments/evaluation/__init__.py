"""
Comprehensive Evaluation Framework for DreamerV3 vs hDreamer Comparison

This module provides systematic evaluation tools for comparing DreamerV3 and hDreamer
across multiple dimensions:

1. Task Performance - Episode returns, success rates, convergence speed
2. Reconstruction Quality - Pixel-wise MSE, perceptual loss, visual quality
3. Sample Efficiency - Learning curves, data efficiency metrics
4. Multi-task Generalization - Transfer learning, cross-task performance
5. Factor Disentanglement - Linear probing, MI analysis, hierarchical structure

Usage:
    from experiments.evaluation import TaskPerformanceEvaluator, ReconstructionEvaluator
    
    # Evaluate task performance
    perf_eval = TaskPerformanceEvaluator(config)
    results = perf_eval.evaluate_model(model, env)
    
    # Evaluate reconstruction quality
    recon_eval = ReconstructionEvaluator(config)
    metrics = recon_eval.evaluate_reconstruction(model, data)
"""

from .task_performance import TaskPerformanceEvaluator
from .reconstruction import ReconstructionEvaluator
from .sample_efficiency import SampleEfficiencyEvaluator
from .generalization import GeneralizationEvaluator
from .disentanglement import DisentanglementEvaluator

__all__ = [
    'TaskPerformanceEvaluator',
    'ReconstructionEvaluator', 
    'SampleEfficiencyEvaluator',
    'GeneralizationEvaluator',
    'DisentanglementEvaluator'
]

# Evaluation configuration defaults
DEFAULT_EVAL_CONFIG = {
    'num_episodes': 100,
    'num_seeds': 5,
    'confidence_level': 0.95,
    'save_videos': True,
    'save_metrics': True,
    'plot_results': True
}

class EvaluationSuite:
    """
    Comprehensive evaluation suite that runs all evaluation modules.
    
    This is the main interface for systematic model comparison.
    """
    
    def __init__(self, config=None):
        self.config = config or DEFAULT_EVAL_CONFIG
        
        # Initialize all evaluators
        self.task_evaluator = TaskPerformanceEvaluator(self.config)
        self.recon_evaluator = ReconstructionEvaluator(self.config)
        self.efficiency_evaluator = SampleEfficiencyEvaluator(self.config)
        self.generalization_evaluator = GeneralizationEvaluator(self.config)
        self.disentanglement_evaluator = DisentanglementEvaluator(self.config)
    
    def evaluate_model_pair(self, dreamerv3_model, hdreamer_model, environments, data):
        """
        Comprehensive evaluation of DreamerV3 vs hDreamer model pair.
        
        Args:
            dreamerv3_model: Trained DreamerV3 model
            hdreamer_model: Trained hDreamer model  
            environments: List of evaluation environments
            data: Evaluation dataset
            
        Returns:
            dict: Comprehensive evaluation results
        """
        results = {}
        
        # 1. Task Performance Comparison
        print("Evaluating task performance...")
        results['task_performance'] = self.task_evaluator.compare_models(
            dreamerv3_model, hdreamer_model, environments
        )
        
        # 2. Reconstruction Quality Comparison
        print("Evaluating reconstruction quality...")
        results['reconstruction'] = self.recon_evaluator.compare_models(
            dreamerv3_model, hdreamer_model, data
        )
        
        # 3. Sample Efficiency Comparison
        print("Evaluating sample efficiency...")
        results['sample_efficiency'] = self.efficiency_evaluator.compare_models(
            dreamerv3_model, hdreamer_model, environments
        )
        
        # 4. Generalization Comparison
        print("Evaluating generalization...")
        results['generalization'] = self.generalization_evaluator.compare_models(
            dreamerv3_model, hdreamer_model, environments
        )
        
        # 5. Factor Disentanglement Analysis
        print("Evaluating factor disentanglement...")
        results['disentanglement'] = self.disentanglement_evaluator.compare_models(
            dreamerv3_model, hdreamer_model, data
        )
        
        # Generate comprehensive report
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _generate_summary(self, results):
        """Generate a summary of all evaluation results."""
        summary = {
            'winner_by_metric': {},
            'statistical_significance': {},
            'key_findings': [],
            'recommendations': []
        }
        
        # Determine winner for each metric
        for category, metrics in results.items():
            if category == 'summary':
                continue
                
            for metric_name, values in metrics.items():
                if isinstance(values, dict) and 'dreamerv3' in values and 'hdreamer' in values:
                    dreamerv3_val = values['dreamerv3']
                    hdreamer_val = values['hdreamer']
                    
                    if dreamerv3_val > hdreamer_val:
                        winner = 'dreamerv3'
                    elif hdreamer_val > dreamerv3_val:
                        winner = 'hdreamer'
                    else:
                        winner = 'tie'
                    
                    summary['winner_by_metric'][f"{category}_{metric_name}"] = winner
        
        return summary


def load_evaluation_config(config_path):
    """Load evaluation configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_evaluation_results(results, output_path):
    """Save evaluation results to file."""
    import json
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def plot_comparison_results(results, output_dir):
    """Generate comparison plots from evaluation results."""
    import matplotlib.pyplot as plt
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Task performance comparison
    if 'task_performance' in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Plot task performance metrics
        # Implementation details in task_performance.py
        plt.savefig(os.path.join(output_dir, 'task_performance_comparison.png'))
        plt.close()

    # Add more plotting functions as needed
