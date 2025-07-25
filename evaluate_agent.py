#!/usr/bin/env python3
"""
VizDoom Agent Evaluation and Visualization Script
Visualizes agent behavior, world model predictions, and performance metrics
"""

import argparse
import pathlib
import sys
import yaml
import torch
import numpy as np
import functools
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict
import json

# Add the current directory to Python path for imports
sys.path.append(str(pathlib.Path(__file__).parent))

import tools
import envs
from dreamer import Dreamer, make_dataset, make_env
from tools import Logger
from parallel import Damy


def to_np(value):
    """Convert tensor to numpy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


class AgentEvaluator:
    def __init__(self, logdir, config_name='vizdoom', device='cuda'):
        self.logdir = pathlib.Path(logdir)
        self.device = device
        
        # Load configuration
        configs = yaml.safe_load(
            (pathlib.Path(__file__).parent / "configs.yaml").read_text()
        )

        # Get base config and update with specific config
        def recursive_update(base, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in base:
                    recursive_update(base[key], value)
                else:
                    base[key] = value

        defaults = configs['defaults'].copy()
        if config_name in configs:
            recursive_update(defaults, configs[config_name])

        # Override device
        defaults['device'] = device

        # Convert scientific notation strings to numbers recursively
        def convert_numeric_strings(obj):
            if isinstance(obj, dict):
                return {k: convert_numeric_strings(v) for k, v in obj.items()}
            elif isinstance(obj, str) and ('e' in obj or 'E' in obj):
                try:
                    num = float(obj)
                    return int(num) if num.is_integer() else num
                except ValueError:
                    return obj  # Keep as string if not a number
            else:
                return obj

        defaults = convert_numeric_strings(defaults)

        # Convert to namespace object
        config = argparse.Namespace(**defaults)
        self.config = config
        
        print(f"Evaluating task: {config.task}")
        print(f"Using device: {device}")
        
        # Create environments
        make_env_fn = lambda mode, id: make_env(config, mode, id)
        eval_envs = [make_env_fn("eval", i) for i in range(config.envs)]
        self.eval_envs = [Damy(env) for env in eval_envs]

        # Set num_actions from action space
        acts = self.eval_envs[0].action_space
        config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
        
        # Load dataset for video prediction
        eval_eps = tools.load_episodes(self.logdir / 'eval_eps', limit=1)
        if not eval_eps:
            # Fallback to train episodes if eval episodes not available
            eval_eps = tools.load_episodes(self.logdir / 'train_eps', limit=1)
        
        self.eval_dataset = make_dataset(eval_eps, config)
        
        # Create a dummy logger for agent initialization
        dummy_logger = Logger(self.logdir / 'temp', 0)

        # Initialize agent
        self.agent = Dreamer(
            self.eval_envs[0].observation_space,
            self.eval_envs[0].action_space,
            config,
            dummy_logger,
            self.eval_dataset,
        ).to(device)
        
        # Load trained model
        self._load_model()
        
    def _load_model(self):
        """Load the trained model from checkpoint."""
        checkpoint_path = self.logdir / "latest.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(self.agent, checkpoint["optims_state_dict"])
        self.agent._should_pretrain._once = False
        self.agent.requires_grad_(requires_grad=False)
        self.agent.eval()
        
        print("Model loaded successfully!")
        
    def evaluate_performance(self, num_episodes=10, save_videos=True):
        """Evaluate agent performance over multiple episodes."""
        print(f"\nEvaluating agent performance over {num_episodes} episodes...")
        
        # Create temporary cache and logger for evaluation
        eval_cache = {}
        temp_logdir = self.logdir / 'evaluation_temp'
        temp_logdir.mkdir(exist_ok=True)
        logger = Logger(temp_logdir, 0)
        
        # Run evaluation
        eval_policy = functools.partial(self.agent, training=False)
        tools.simulate(
            eval_policy,
            self.eval_envs,
            eval_cache,
            temp_logdir / 'eval_episodes',
            logger,
            is_eval=True,
            episodes=num_episodes,
        )
        
        # Extract performance metrics
        metrics = self._extract_metrics(temp_logdir / 'metrics.jsonl')
        
        # Save evaluation results
        results = {
            'num_episodes': num_episodes,
            'metrics': metrics,
            'config': {
                'task': self.config.task,
                'model_path': str(self.logdir),
            }
        }
        
        results_path = self.logdir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nEvaluation Results:")
        print(f"Average Return: {metrics.get('eval_return', 'N/A'):.2f}")
        print(f"Average Length: {metrics.get('eval_length', 'N/A'):.1f}")
        print(f"Results saved to: {results_path}")
        
        return results
        
    def _extract_metrics(self, metrics_file):
        """Extract metrics from jsonl file."""
        metrics = {}
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    metrics.update(data)
        return metrics
        
    def generate_video_predictions(self, num_sequences=3, save_path=None):
        """Generate world model video predictions."""
        print(f"\nGenerating {num_sequences} video prediction sequences...")
        
        predictions = []
        for i in range(num_sequences):
            try:
                data_batch = next(self.eval_dataset)
                video_pred = self.agent._wm.video_pred(data_batch)
                video_pred_np = to_np(video_pred)
                predictions.append(video_pred_np)
                print(f"Generated prediction {i+1}/{num_sequences}")
            except Exception as e:
                print(f"Error generating prediction {i+1}: {e}")
                continue
        
        if save_path is None:
            save_path = self.logdir / 'video_predictions.npz'
        
        if predictions:
            np.savez_compressed(save_path, *predictions)
            print(f"Video predictions saved to: {save_path}")
        
        return predictions
        
    def create_comparison_video(self, predictions, output_path=None, fps=10):
        """Create side-by-side comparison video of truth vs model predictions."""
        if not predictions:
            print("No predictions available for comparison video")
            return

        if output_path is None:
            output_path = self.logdir / 'comparison_video.mp4'

        print(f"Creating comparison video: {output_path}")

        # Use first prediction sequence
        video_data = predictions[0]  # Shape: [batch, time, height, width*3, channels]

        # Extract truth, model, and error from concatenated video
        batch_size, time_steps, height, width_combined, channels = video_data.shape
        width = width_combined // 3  # truth, model, error are concatenated horizontally

        # Split the concatenated video
        truth = video_data[:, :, :, :width, :]
        model = video_data[:, :, :, width:2*width, :]
        error = video_data[:, :, :, 2*width:, :]

        # Create figure for animation
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].set_title('Ground Truth')
        axes[1].set_title('Model Prediction')
        axes[2].set_title('Prediction Error')

        for ax in axes:
            ax.axis('off')

        # Initialize image displays
        im1 = axes[0].imshow(truth[0, 0], vmin=0, vmax=1)
        im2 = axes[1].imshow(model[0, 0], vmin=0, vmax=1)
        im3 = axes[2].imshow(error[0, 0], vmin=0, vmax=1)

        def animate(frame):
            im1.set_array(truth[0, frame])
            im2.set_array(model[0, frame])
            im3.set_array(error[0, frame])
            return [im1, im2, im3]

        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=time_steps, interval=1000//fps, blit=True
        )

        # Save animation
        try:
            anim.save(output_path, writer='ffmpeg', fps=fps)
            print(f"Comparison video saved to: {output_path}")
        except Exception as e:
            print(f"Error saving video: {e}")
            print("Make sure ffmpeg is installed for video saving")

        plt.close(fig)

    def analyze_training_progress(self):
        """Analyze training progress from metrics file."""
        metrics_file = self.logdir / 'metrics.jsonl'
        if not metrics_file.exists():
            print("No training metrics found")
            return None

        print("Analyzing training progress...")

        # Read all metrics
        metrics_data = []
        with open(metrics_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    metrics_data.append(data)
                except:
                    continue

        if not metrics_data:
            print("No valid metrics data found")
            return None

        # Create training progress plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Training Progress - {self.config.task}')

        steps = [d.get('step', 0) for d in metrics_data]

        # Plot training return
        train_returns = [d.get('train_return') for d in metrics_data if 'train_return' in d]
        train_steps = [d.get('step', 0) for d in metrics_data if 'train_return' in d]
        if train_returns:
            axes[0, 0].plot(train_steps, train_returns)
            axes[0, 0].set_title('Training Return')
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Return')

        # Plot evaluation return
        eval_returns = [d.get('eval_return') for d in metrics_data if 'eval_return' in d]
        eval_steps = [d.get('step', 0) for d in metrics_data if 'eval_return' in d]
        if eval_returns:
            axes[0, 1].plot(eval_steps, eval_returns, 'orange')
            axes[0, 1].set_title('Evaluation Return')
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('Return')

        # Plot episode length
        train_lengths = [d.get('train_length') for d in metrics_data if 'train_length' in d]
        if train_lengths:
            axes[1, 0].plot(train_steps[:len(train_lengths)], train_lengths)
            axes[1, 0].set_title('Episode Length')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Length')

        # Plot world model loss
        wm_losses = [d.get('dyn_loss') for d in metrics_data if 'dyn_loss' in d]
        wm_steps = [d.get('step', 0) for d in metrics_data if 'dyn_loss' in d]
        if wm_losses:
            axes[1, 1].plot(wm_steps, wm_losses)
            axes[1, 1].set_title('World Model Loss')
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Loss')

        plt.tight_layout()

        # Save plot
        plot_path = self.logdir / 'training_progress.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Training progress plot saved to: {plot_path}")
        plt.close(fig)

        return metrics_data

    def create_action_analysis(self, num_episodes=5):
        """Analyze action distribution during evaluation."""
        print(f"Analyzing action distribution over {num_episodes} episodes...")

        action_counts = defaultdict(int)
        total_steps = 0

        # Run episodes and collect action statistics
        for episode in range(num_episodes):
            obs = self.eval_envs[0].reset()
            done = False
            episode_steps = 0

            while not done and episode_steps < 1000:  # Limit episode length
                # Get action from agent
                obs_tensor = {k: torch.tensor(v).unsqueeze(0).to(self.device)
                             for k, v in obs.items()}
                reset_tensor = torch.tensor([False]).to(self.device)

                with torch.no_grad():
                    policy_output, _ = self.agent(obs_tensor, reset_tensor, training=False)
                    action = policy_output["action"].cpu().numpy()[0]

                action_counts[action] += 1
                total_steps += 1
                episode_steps += 1

                # Step environment
                obs, reward, done, info = self.eval_envs[0].step(action)

        # Create action distribution plot
        if action_counts:
            actions = list(action_counts.keys())
            counts = list(action_counts.values())

            plt.figure(figsize=(10, 6))
            plt.bar(actions, counts)
            plt.title(f'Action Distribution - {self.config.task}')
            plt.xlabel('Action')
            plt.ylabel('Count')

            # Add action labels if available
            if hasattr(self.eval_envs[0], 'actions'):
                action_labels = [f"Action {i}" for i in actions]
                plt.xticks(actions, action_labels, rotation=45)

            plt.tight_layout()

            # Save plot
            plot_path = self.logdir / 'action_distribution.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Action distribution plot saved to: {plot_path}")
            plt.close()

            # Print statistics
            print(f"Total steps analyzed: {total_steps}")
            for action, count in sorted(action_counts.items()):
                percentage = (count / total_steps) * 100
                print(f"Action {action}: {count} ({percentage:.1f}%)")

        return action_counts


def main():
    parser = argparse.ArgumentParser(description='Evaluate VizDoom DreamerV3 Agent')
    parser.add_argument('--logdir', type=str, required=True,
                       help='Path to model checkpoint directory')
    parser.add_argument('--config', type=str, default='vizdoom',
                       help='Configuration name to use')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--video-pred', action='store_true',
                       help='Generate video predictions')
    parser.add_argument('--comparison-video', action='store_true',
                       help='Create comparison video')
    parser.add_argument('--training-analysis', action='store_true',
                       help='Analyze training progress')
    parser.add_argument('--action-analysis', action='store_true',
                       help='Analyze action distribution')
    parser.add_argument('--all', action='store_true',
                       help='Run all analyses')

    args = parser.parse_args()

    # Initialize evaluator
    try:
        evaluator = AgentEvaluator(args.logdir, args.config, args.device)
    except Exception as e:
        print(f"Error initializing evaluator: {e}")
        return

    # Run performance evaluation
    try:
        results = evaluator.evaluate_performance(args.episodes)
    except Exception as e:
        print(f"Error during performance evaluation: {e}")
        results = None

    # Generate video predictions if requested
    if args.video_pred or args.comparison_video or args.all:
        try:
            predictions = evaluator.generate_video_predictions()

            if (args.comparison_video or args.all) and predictions:
                evaluator.create_comparison_video(predictions)
        except Exception as e:
            print(f"Error generating video predictions: {e}")

    # Analyze training progress if requested
    if args.training_analysis or args.all:
        try:
            evaluator.analyze_training_progress()
        except Exception as e:
            print(f"Error analyzing training progress: {e}")

    # Analyze action distribution if requested
    if args.action_analysis or args.all:
        try:
            evaluator.create_action_analysis()
        except Exception as e:
            print(f"Error analyzing actions: {e}")

    print("\nEvaluation complete!")
    if results:
        print(f"Results saved in: {evaluator.logdir}")


if __name__ == "__main__":
    main()
