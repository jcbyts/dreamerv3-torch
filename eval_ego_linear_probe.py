#!/usr/bin/env python3
"""
Ego motion linear probing evaluation for VizDoom world model.
Tests if ego-motion factors can be linearly decoded from learned latent representations.
Supports both hierarchical (top/bottom/combined) and flat (all) models.
"""

import argparse
import pathlib
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter

from evaluate_agent import AgentEvaluator
from enhanced_vizdoom import EnhancedViZDoom, extract_ground_truth_factors

# Ego-motion factors to evaluate
EGO_FACTORS = [
    "ego_delta_x", "ego_delta_y", "ego_delta_z",
    "ego_delta_angle_sin", "ego_delta_angle_cos",   # derived from ego_delta_angle
    "ego_vel_x", "ego_vel_y", "ego_vel_z",
]


def _collect_data(evaluator, env, device, episodes, max_steps):
    """Collect latent representations and ground truth factors from episodes."""
    latents = []
    factors = []
    
    for episode in range(episodes):
        obs = env.reset()
        done = False
        step_count = 0

        # Initialize agent and latent states
        agent_state = None
        latent_state = None
        action = None
        prev_game_vars = obs.get('game_variables', None)

        while not done and step_count < max_steps:
            # Build observation tensor (only what's required by agent/wm)
            is_first = torch.tensor([step_count == 0]).to(device)
            is_terminal = torch.tensor([done]).to(device)
            obs_tensor = {
                "image": torch.tensor(obs["image"]).unsqueeze(0).to(device),
                "is_first": is_first,
                "is_terminal": is_terminal
            }
            
            with torch.no_grad():
                wm = evaluator.agent._wm
                
                # Get latent state from world model
                obs_processed = wm.preprocess(obs_tensor)
                embed = wm.encoder(obs_processed)
                latent_state, _ = wm.dynamics.obs_step(
                    latent_state, action, embed, is_first
                )
                
                # Get feature representation
                feat = wm.dynamics.get_feat(latent_state)  # [B, D]
                
                # Get action for next step
                policy_output, agent_state = evaluator.agent(
                    obs_tensor, is_first, agent_state, training=False
                )
                action = policy_output["action"]  # [B, A]
            
            # Store latent vector
            latents.append(feat.squeeze(0).detach().cpu().numpy())
            
            # Extract ground truth factors with previous state for deltas
            raw_factors = extract_ground_truth_factors(obs, prev_game_vars)

            # Update previous state for next iteration
            if 'game_variables' in obs:
                prev_game_vars = obs['game_variables']

            # Convert ego_delta_angle to sin/cos representation
            angle = raw_factors.pop("ego_delta_angle", 0.0)
            raw_factors["ego_delta_angle_sin"] = np.sin(angle)
            raw_factors["ego_delta_angle_cos"] = np.cos(angle)
            
            # Add episode ID for train/test splitting
            raw_factors["_episode_id"] = episode
            factors.append(raw_factors)
            
            # Step environment - convert action to integer index
            action_np = action.cpu().numpy()[0]
            if len(action_np.shape) > 0 and action_np.shape[0] > 1:
                # One-hot action, convert to index
                action_idx = int(np.argmax(action_np))
            else:
                # Already scalar action
                action_idx = int(action_np)
            obs, reward, done, info = env.step(action_idx)
            step_count += 1
    
    return np.asarray(latents), factors


def _get_latent_blocks(latents, evaluator):
    """Split latents into hierarchical blocks or return all latents for flat models."""
    wm = evaluator.agent._wm
    dyn = wm.dynamics
    
    # Detect hierarchical mode robustly
    hierarchical = (hasattr(dyn, "_hierarchical_mode") and
                   getattr(dyn, "_hierarchical_mode", False) and
                   hasattr(dyn, "_stoch_top") and hasattr(dyn, "_stoch_bottom") and
                   getattr(dyn, "_stoch_bottom", 0) > 0)
    
    if hierarchical:
        # Get hierarchical dimensions
        z_top_dim = dyn._stoch_top * dyn._discrete_top
        z_bottom_dim = dyn._stoch_bottom * dyn._discrete_bottom
        
        # Verify latent dimensions match expectations
        expected_min_dim = z_top_dim + z_bottom_dim
        assert latents.shape[1] >= expected_min_dim, (
            f"Latent dims {latents.shape[1]} smaller than expected {expected_min_dim}"
        )
        
        # Split latents: [z_top_flat, z_bottom_flat, h]
        X_top = latents[:, :z_top_dim]
        X_bottom = latents[:, z_top_dim:z_top_dim + z_bottom_dim]
        X_combined = latents  # includes [z_top, z_bottom, h]
        
        return hierarchical, {
            "top": X_top,
            "bottom": X_bottom, 
            "combined": X_combined
        }
    else:
        # Flat model: [z_flat, h]
        return hierarchical, {"all": latents}


def _create_episode_split(factors, episodes_train, episodes_test):
    """Create episode-level train/test split to avoid temporal leakage."""
    episode_ids = np.array([f["_episode_id"] for f in factors])
    
    # Define train and test episode sets
    train_episode_ids = set(range(episodes_train))
    test_episode_ids = set(range(episodes_train, episodes_train + episodes_test))
    
    # Create boolean masks
    train_mask = np.isin(episode_ids, list(train_episode_ids))
    test_mask = np.isin(episode_ids, list(test_episode_ids))
    
    return train_mask, test_mask


def _extract_targets(factors, mask, factor_name):
    """Extract target values for a specific factor."""
    return np.array([f[factor_name] for f in factors])[mask]


def _probe_latent_block(X_train, X_test, factors, train_mask, test_mask, log_per_factor):
    """Run linear probing on a specific latent block."""
    per_factor_r2 = {}
    valid_r2_scores = []
    
    for factor_name in EGO_FACTORS:
        # Extract targets
        y_train = _extract_targets(factors, train_mask, factor_name)
        y_test = _extract_targets(factors, test_mask, factor_name)
        
        # Skip factors with no variation
        if np.var(y_train) < 1e-6:
            continue
        
        # Standardize targets
        y_scaler = StandardScaler()
        y_train_std = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_std = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
        
        # Train linear regressor
        regressor = LinearRegression()
        regressor.fit(X_train, y_train_std)
        y_pred_std = regressor.predict(X_test)
        
        # Compute R² score
        r2 = r2_score(y_test_std, y_pred_std)
        r2_clipped = float(max(r2, 0.0))  # Clip negative R² for stability
        
        per_factor_r2[factor_name] = r2_clipped
        valid_r2_scores.append(r2_clipped)
    
    # Compute macro average R²
    macro_r2 = float(np.mean(valid_r2_scores)) if valid_r2_scores else 0.0
    
    # Only keep per-factor results if requested
    if not log_per_factor:
        per_factor_r2 = {}
    
    return macro_r2, per_factor_r2


def run_ego_probe(model_path, config_name, episodes_train, episodes_test,
                  max_steps_per_episode, device='cuda'):
    """
    Clean offline ego motion linear probing evaluation.

    Args:
        model_path: Path to trained model directory
        config_name: Configuration name (e.g., 'vizdoom_health_gathering')
        episodes_train: Number of episodes for training linear probes
        episodes_test: Number of episodes for testing linear probes
        max_steps_per_episode: Maximum steps per episode
        device: Device to use for evaluation

    Returns:
        Dict with keys:
            'hierarchical': bool,
            'r2_total': float,
            'r2_position': float,
            'r2_velocity': float,
            'r2_angle': float,
            'scopes': dict with detailed per-scope results
    """
    model_path = pathlib.Path(model_path)
    task = config_name.replace("vizdoom_", "")
    
    # Initialize evaluator and environment
    evaluator = AgentEvaluator(model_path, config_name, device)
    env = EnhancedViZDoom(task, enable_labels=True, enable_depth=True)
    
    try:
        with torch.no_grad():
            evaluator.agent.eval()
            
            # Collect data from episodes
            total_episodes = episodes_train + episodes_test
            latents, factors = _collect_data(
                evaluator, env, device, total_episodes, max_steps_per_episode
            )
            
            # Get latent block views (hierarchical or flat)
            hierarchical, latent_blocks = _get_latent_blocks(latents, evaluator)
            
            # Create episode-level train/test split
            train_mask, test_mask = _create_episode_split(
                factors, episodes_train, episodes_test
            )

            # Run probing for each latent block
            results = {"hierarchical": hierarchical, "scopes": {}}
            
            for scope_name, X in latent_blocks.items():
                X_train = X[train_mask]
                X_test = X[test_mask]

                macro_r2, per_factor_r2 = _probe_latent_block(
                    X_train, X_test, factors, train_mask, test_mask, False
                )

                # Store results
                results["scopes"][scope_name] = {
                    "macro_r2": macro_r2,
                    "per_factor": per_factor_r2
                }

            # Flatten probe output into the four summary keys
            block = results['scopes'].get('combined') or results['scopes'].get('all')
            if block:
                results['r2_total'] = float(block['macro_r2'])
                results['r2_position'] = float(block['per_factor'].get('ego_delta_x', 0.0))

                # Average velocity R² across x, y, z components
                vel = ['ego_vel_x', 'ego_vel_y', 'ego_vel_z']
                results['r2_velocity'] = float(sum(block['per_factor'].get(f, 0.0) for f in vel) / 3)

                # Average angle R² across sin/cos components
                a_s = block['per_factor'].get('ego_delta_angle_sin', 0.0)
                a_c = block['per_factor'].get('ego_delta_angle_cos', 0.0)
                results['r2_angle'] = float((a_s + a_c) / 2)
            else:
                # Fallback if no primary block found
                results['r2_total'] = 0.0
                results['r2_position'] = 0.0
                results['r2_velocity'] = 0.0
                results['r2_angle'] = 0.0
            
    finally:
        env.close()
    
    return results


def main():
    """Command line interface for ego motion linear probing."""
    parser = argparse.ArgumentParser(
        description='Ego motion linear probing evaluation for VizDoom world model'
    )
    parser.add_argument('--model-path', required=True, type=str,
                       help='Path to trained model directory')
    parser.add_argument('--config', required=True, type=str,
                       help='Configuration name (e.g., vizdoom_basic)')
    parser.add_argument('--step', required=True, type=int,
                       help='Current training step for tensorboard x-axis')
    parser.add_argument('--episodes-train', type=int, default=6,
                       help='Number of episodes for training linear probes')
    parser.add_argument('--episodes-test', type=int, default=4,
                       help='Number of episodes for testing linear probes')
    parser.add_argument('--max-steps-per-episode', type=int, default=300,
                       help='Maximum steps per episode')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for evaluation')
    parser.add_argument('--tb-dir', type=str, default=None,
                       help='Tensorboard directory (defaults to model-path/linear_probe_tb)')
    parser.add_argument('--log-per-factor', action='store_true',
                       help='Log per-factor R² scores to tensorboard')
    
    args = parser.parse_args()
    
    # Run ego motion probing
    results = run_ego_probe(
        model_path=args.model_path,
        config_name=args.config,
        step=args.step,
        device=args.device,
        episodes_train=args.episodes_train,
        episodes_test=args.episodes_test,
        max_steps_per_episode=args.max_steps_per_episode,
        log_per_factor=args.log_per_factor,
        tb_dir=args.tb_dir,
    )
    
    # Print summary
    print(f"\nEgo Motion Linear Probing Results (Step {args.step}):")
    print(f"Model: {'Hierarchical' if results['hierarchical'] else 'Flat'}")
    
    for scope_name, scope_results in results["scopes"].items():
        macro_r2 = scope_results["macro_r2"]
        print(f"  {scope_name.capitalize()}: Macro R² = {macro_r2:.3f}")
        
        if scope_results["per_factor"]:
            for factor_name, r2_score in scope_results["per_factor"].items():
                print(f"    {factor_name}: R² = {r2_score:.3f}")


if __name__ == "__main__":
    main()
