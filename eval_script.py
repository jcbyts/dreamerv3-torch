#!/usr/bin/env python3
"""
Interactive Evaluation Script for DreamerV3/hDreamer Linear Probing
Use with VS Code or Jupyter for cell-by-cell execution with #%% markers
"""

#%% Imports and Setup
import numpy as np
import torch
import matplotlib.pyplot as plt
import pathlib
from collections import defaultdict
import json

# Local imports
from evaluate_agent import AgentEvaluator
from enhanced_vizdoom import EnhancedViZDoom, extract_ground_truth_factors
from eval_ego_linear_probe import run_ego_probe, _collect_data, _get_latent_blocks, _create_episode_split, _probe_latent_block

# Set up matplotlib for interactive plotting
plt.ion()  # Turn on interactive mode
plt.style.use('default')

print("🔬 Interactive Evaluation Script Loaded")
print("Available functions:")
print("  - AgentEvaluator: Load and evaluate models")
print("  - EnhancedViZDoom: VizDoom environment with ground truth factors")
print("  - run_ego_probe: Complete linear probing pipeline")
print("  - Individual functions: _collect_data, _get_latent_blocks, etc.")

#%% Configuration
# Model paths - modify these for your specific models
DREAMERV3_MODEL = "./logdir/comparison_20250727_114209/dreamerv3_seed_1"
HDREAMER_MODEL = "./logdir/comparison_20250727_114209/hdreamer_seed_1"
CONFIG_NAME = "vizdoom_health_gathering"
DEVICE = "cuda:0"

print(f"📁 DreamerV3 model: {DREAMERV3_MODEL}")
print(f"📁 hDreamer model: {HDREAMER_MODEL}")
print(f"⚙️  Config: {CONFIG_NAME}")
print(f"🖥️  Device: {DEVICE}")

#%% Load a Model (Choose one)
# Load DreamerV3 model
model_path = DREAMERV3_MODEL
model_path = HDREAMER_MODEL  # Uncomment to load hDreamer instead

print(f"🔄 Loading model from: {model_path}")
evaluator = AgentEvaluator(model_path, CONFIG_NAME, DEVICE)
print("✅ Model loaded successfully!")

# Check model architecture
dyn = evaluator.agent._wm.dynamics
print(f"\n🏗️  Model Architecture:")
print(f"  Hierarchical mode: {getattr(dyn, '_hierarchical_mode', False)}")
print(f"  Stoch top: {getattr(dyn, '_stoch_top', 'N/A')}")
print(f"  Stoch bottom: {getattr(dyn, '_stoch_bottom', 'N/A')}")
print(f"  Discrete top: {getattr(dyn, '_discrete_top', 'N/A')}")
print(f"  Discrete bottom: {getattr(dyn, '_discrete_bottom', 'N/A')}")

#%% Initialize Environment
print("🎮 Initializing VizDoom environment...")
env = EnhancedViZDoom('health_gathering', enable_labels=True, enable_depth=True)
print(f"✅ Environment initialized")
print(f"📊 Available game variables: {env._game_var_names}")

#%% Single Episode Data Collection
print("🔍 Running single episode for debugging...")

# Reset environment
obs = env.reset()
print(f"🔄 Environment reset")
print(f"📸 Image shape: {obs['image'].shape}")
print(f"🎯 Game variables: {obs.get('game_variables', 'None')}")

# Initialize agent state
agent_state = None
latent_state = None
action = None
prev_game_vars = obs.get('game_variables', None)

# Collect data for a few steps
latents_debug = []
factors_debug = []
images_debug = []

max_steps = 1020
step_count = 0
done = False

print(f"🏃 Collecting {max_steps} steps of data...")

while not done and step_count < max_steps:
    # Store image for visualization
    images_debug.append(obs['image'].copy())
    
    # Build observation tensor
    is_first = torch.tensor([step_count == 0]).to(DEVICE)
    is_terminal = torch.tensor([done]).to(DEVICE)
    obs_tensor = {
        "image": torch.tensor(obs["image"]).unsqueeze(0).to(DEVICE),
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
    latents_debug.append(feat.squeeze(0).detach().cpu().numpy())
    
    # Extract ground truth factors
    raw_factors = extract_ground_truth_factors(obs, prev_game_vars)
    
    # Update previous state
    if 'game_variables' in obs:
        prev_game_vars = obs['game_variables']
    
    # Convert ego_delta_angle to sin/cos representation
    angle = raw_factors.pop("ego_delta_angle", 0.0)
    raw_factors["ego_delta_angle_sin"] = np.sin(angle)
    raw_factors["ego_delta_angle_cos"] = np.cos(angle)
    
    # Add step info
    raw_factors["_step"] = step_count
    factors_debug.append(raw_factors)
    
    # Step environment
    action_np = action.cpu().numpy()[0]
    if len(action_np.shape) > 0 and action_np.shape[0] > 1:
        action_idx = int(np.argmax(action_np))
    else:
        action_idx = int(action_np)
    
    obs, reward, done, info = env.step(action_idx)
    step_count += 1
    
    print(f"  Step {step_count}: action={action_idx}, reward={reward:.1f}")

print(f"✅ Collected {len(latents_debug)} steps of data")
print(f"📊 Latent shape: {latents_debug[0].shape}")
print(f"🎯 Factor keys: {list(factors_debug[0].keys())}")

#%% Visualize Images
print("🖼️  Visualizing collected images...")

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Collected VizDoom Images', fontsize=14)

for i in range(min(10, len(images_debug))):
    row = i // 5
    col = i % 5
    axes[row, col].imshow(images_debug[i])
    axes[row, col].set_title(f'Step {i}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

#%% Analyze Ground Truth Factors
print("📈 Analyzing ground truth factors...")

# Convert factors to arrays for analysis
factor_names = [k for k in factors_debug[0].keys() if not k.startswith('_')]
factor_data = {}

for name in factor_names:
    values = [f.get(name, 0.0) for f in factors_debug]
    factor_data[name] = np.array(values)
    print(f"  {name}: {np.mean(values):.3f} ± {np.std(values):.3f} (range: {np.min(values):.3f} to {np.max(values):.3f})")

# Plot key ego motion factors
ego_factors = ['ego_pos_x', 'ego_pos_y', 'ego_vel_x', 'ego_vel_y', 'ego_delta_x', 'ego_delta_y']
available_ego = [f for f in ego_factors if f in factor_data]

if available_ego:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Ego Motion Factors Over Time', fontsize=14)
    
    for i, factor in enumerate(available_ego[:6]):
        row = i // 3
        col = i % 3
        axes[row, col].plot(factor_data[factor], 'o-')
        axes[row, col].set_title(factor)
        axes[row, col].set_xlabel('Step')
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

#%% Analyze Latent Representations
print("🧠 Analyzing latent representations...")

latents_array = np.array(latents_debug)
print(f"📊 Latents shape: {latents_array.shape}")

# Get latent blocks (hierarchical vs flat)
hierarchical, latent_blocks = _get_latent_blocks(latents_array, evaluator)
print(f"🏗️  Hierarchical: {hierarchical}")

for name, block in latent_blocks.items():
    print(f"  {name}: {block.shape}")
    print(f"    Mean: {np.mean(block):.3f}, Std: {np.std(block):.3f}")

# Visualize latent activations
fig, axes = plt.subplots(1, len(latent_blocks), figsize=(5*len(latent_blocks), 4))
if len(latent_blocks) == 1:
    axes = [axes]

for i, (name, block) in enumerate(latent_blocks.items()):
    # Show first 50 dimensions over time
    dims_to_show = min(150, block.shape[1])
    im = axes[i].imshow(block[:, :dims_to_show].T, aspect='auto', cmap='viridis')
    axes[i].set_title(f'{name} Latents\n({block.shape[1]} dims)')
    axes[i].set_xlabel('Time Step')
    axes[i].set_ylabel('Latent Dimension')
    plt.colorbar(im, ax=axes[i])

plt.tight_layout()
plt.show()

#%% Clean up
env.close()
print("🧹 Environment closed")
print("✅ Debug session complete!")
print("\n💡 Next steps:")
print("  - Modify model_path to test different models")
print("  - Adjust max_steps for longer episodes")
print("  - Run full linear probing with run_ego_probe()")
print("  - Compare latent representations between DreamerV3 and hDreamer")
