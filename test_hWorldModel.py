#!/usr/bin/env python3
"""
Interactive test script for hWorldModel with VizDoom Deadly Corridor.
Run this in Jupyter or step through cells in VS Code.
"""
#%%
import sys
import pathlib
import argparse
import numpy as np
import torch
import ruamel.yaml as yaml
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(str(pathlib.Path(__file__).parent))

import models
import tools
#%%
from envs.vizdoom import ViZDoom

#%% Cell 1: Load Configuration
print("=== Loading Configuration ===")

# Load configs
configs = yaml.safe_load(
    (pathlib.Path(__file__).parent / "configs.yaml").read_text()
)

def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value

# Merge configs: defaults + hdreamer + vizdoom_deadly_corridor
name_list = ["defaults", "hdreamer", "vizdoom_deadly_corridor"]
defaults = {}
for name in name_list:
    recursive_update(defaults, configs[name])

# Convert to namespace
parser = argparse.ArgumentParser()
for key, value in sorted(defaults.items(), key=lambda x: x[0]):
    arg_type = tools.args_type(value)
    parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

config = parser.parse_args([])
config._config_names = ["hdreamer", "vizdoom_deadly_corridor"]

print(f"✅ Config loaded")
print(f"   Model: {config.model}")
print(f"   Task: {config.task}")
print(f"   Device: {config.device}")
print(f"   h_levels: {config.rssm['h_levels']}")
print(f"   h_stoch_dims: {config.rssm['h_stoch_dims']}")
print(f"   h_deter_dims: {config.rssm['h_deter_dims']}")

#%% Cell 2: Create Environment
print("\n=== Creating Environment ===")

env = ViZDoom('deadly_corridor')
print(f"✅ Environment created")
print(f"   Observation space: {env.observation_space}")
print(f"   Action space: {env.action_space}")

# Set num_actions from action space (same as dreamer.py)
config.num_actions = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
print(f"   num_actions: {config.num_actions}")

#%% Cell 3: Test Environment
print("\n=== Testing Environment ===")

obs = env.reset()
print(f"✅ Environment reset")
print(f"   Observation keys: {list(obs.keys())}")
print(f"   Image shape: {obs['image'].shape}")
print(f"   Image dtype: {obs['image'].dtype}")
print(f"   Image range: [{obs['image'].min()}, {obs['image'].max()}]")

# Take a random action
action = env.action_space.sample()
obs, reward, done, info = env.step(action)
print(f"✅ Environment step")
print(f"   Action: {action}")
print(f"   Reward: {reward}")
print(f"   Done: {done}")

#%% Cell 4: Create hWorldModel
print("\n=== Creating hWorldModel ===")

obs_space = env.observation_space
act_space = env.action_space


step = 0

try:
    world_model = models.hWorldModel(obs_space, act_space, step, config)
    world_model = world_model.to(config.device)  # Move model to correct device
    print(f"✅ hWorldModel created successfully")
    print(f"   Device: {config.device}")
    print(f"   Encoder type: {type(world_model.encoder).__name__}")
    print(f"   Dynamics type: {type(world_model.dynamics).__name__}")
    print(f"   Heads: {list(world_model.heads.keys())}")
except Exception as e:
    print(f"❌ hWorldModel creation failed: {e}")
    import traceback
    traceback.print_exc()

#%% Cell 5: Test Encoder
print("\n=== Testing Encoder ===")

# Prepare observation dictionary for encoder (like preprocess does)
obs_dict = {
    'image':          torch.from_numpy(obs['image']).float()
                      .unsqueeze(0).unsqueeze(0),            # (B=1, T=1, H, W, C)
    'is_first':       torch.tensor([[True]], dtype=torch.bool),
    'is_terminal':    torch.tensor([[False]], dtype=torch.bool)
}
# Preprocess the observation (this normalizes image and adds required fields)
obs_processed = world_model.preprocess(obs_dict)

print(f"Processed obs keys: {list(obs_processed.keys())}")
print(f"Image shape: {obs_processed['image'].shape}")
print(f"Image range: [{obs_processed['image'].min():.3f}, {obs_processed['image'].max():.3f}]")
print(f"Image device: {obs_processed['image'].device}")
print(f"Model device: {next(world_model.parameters()).device}")

try:
    embed_list = world_model.encoder(obs_processed)
    print(f"✅ Encoder forward pass successful")
    print(f"   Number of levels: {len(embed_list)}")
    for i, embed in enumerate(embed_list):
        print(f"   Level {i} shape: {embed.shape}")
except Exception as e:
    print(f"❌ Encoder forward pass failed: {e}")
    import traceback
    traceback.print_exc()

#%% Cell 6: Test Dynamics (obs_step)
print("\n=== Testing Dynamics obs_step ===")

if 'embed_list' in locals():
    try:
        # Prepare inputs for dynamics
        
        to_dev = lambda x: x.to(config.device)
        prev_state = None
        prev_action = to_dev(torch.zeros((1, config.num_actions)))   # (B, A)
        is_first    = torch.ones((1,), dtype=torch.bool, device=config.device)

        print(f"Action tensor shape: {prev_action.shape}")
        print(f"is_first shape: {is_first.shape}")

        # Call obs_step
        post, prior, spatial_post = world_model.dynamics.obs_step(
            prev_state, prev_action, embed_list, is_first, sample=True
        )

        print(f"✅ Dynamics obs_step successful")
        print(f"   Post keys: {list(post.keys())}")
        print(f"   Prior keys: {list(prior.keys())}")
        print(f"   Spatial post shape: {spatial_post.shape}")

        # Check hierarchical structure
        if 'stoch' in post and isinstance(post['stoch'], list):
            print(f"   Hierarchical stoch levels: {len(post['stoch'])}")
            for i, stoch in enumerate(post['stoch']):
                print(f"     Level {i} stoch shape: {stoch.shape}")
        
    except Exception as e:
        print(f"❌ Dynamics obs_step failed: {e}")
        import traceback
        traceback.print_exc()

#%% Cell 7: Test Decoder
print("\n=== Testing Decoder ===")

if 'spatial_post' in locals():
    try:
        decoder_output = world_model.heads['decoder'](spatial_post)
        print(f"✅ Decoder forward pass successful")
        print(f"   Decoder output keys: {list(decoder_output.keys())}")
        
        if 'image' in decoder_output:
            recon = decoder_output['image'].mode()
            print(f"   Reconstruction shape: {recon.shape}")
            print(f"   Reconstruction range: [{recon.min():.3f}, {recon.max():.3f}]")
        
    except Exception as e:
        print(f"❌ Decoder forward pass failed: {e}")
        import traceback
        traceback.print_exc()

#%% Cell 8: Test Full Pipeline
print("\n=== Testing Full Pipeline ===")

try:
    # Simulate a full forward pass like in dreamer.py
    obs_dict_full = {
        'image': torch.from_numpy(obs['image']).float().unsqueeze(0).unsqueeze(0),
        'is_first': torch.ones((1, 1), dtype=torch.bool),
        'is_terminal': torch.zeros((1, 1), dtype=torch.bool)
    }
    
    # Preprocess (like in WorldModel.preprocess)
    obs_processed = world_model.preprocess(obs_dict_full)
    print(f"✅ Preprocessing successful")
    print(f"   Processed obs keys: {list(obs_processed.keys())}")
    
    # Encode
    embed_list = world_model.encoder(obs_processed)
    print(f"✅ Encoding successful")
    
    # Dynamics
    post, prior, spatial_post = world_model.dynamics.obs_step(
        None, torch.zeros((1, config.num_actions)), embed_list, 
        obs_processed['is_first'].squeeze(-1), sample=True
    )
    print(f"✅ Dynamics successful")
    
    # Decode
    decoder_output = world_model.heads['decoder'](spatial_post)
    print(f"✅ Full pipeline successful!")
    
except Exception as e:
    print(f"❌ Full pipeline failed: {e}")
    import traceback
    traceback.print_exc()

#%% Cell 9: Summary
print("\n=== Summary ===")
print("If all tests passed, your hWorldModel is working correctly!")
print("You can now use it in the full training loop.")

# Clean up
env.close()

# %%
