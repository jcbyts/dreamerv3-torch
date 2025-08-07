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

#%% Cell 9: Full hWorldModel backward test with larger batch
print("\n=== Testing full hWorldModel backward pass (batch size 4) ===")

import torch

# turn on anomaly detection just for this block
torch.autograd.set_detect_anomaly(True)

try:
    batch_size = 4

    # 1) Prepare a batch by repeating the single obs 4×
    img = torch.from_numpy(obs['image']).float()  # (H,W,C)
    img = img.unsqueeze(0).unsqueeze(0)           # (1,1,H,W,C)
    img = img.expand(batch_size, 1, *img.shape[2:]).to(config.device)

    obs_batch = {
        'image':       img,
        'is_first':    torch.ones((batch_size,1), dtype=torch.bool, device=config.device),
        'is_terminal': torch.zeros((batch_size,1), dtype=torch.bool, device=config.device),
    }

    # 2) Preprocess & encode
    proc   = world_model.preprocess(obs_batch)
    embeds = world_model.encoder(proc)

    # 3) Dynamics
    prev_action = torch.zeros((batch_size, config.num_actions), device=config.device, requires_grad=True)
    post, prior, spat = world_model.dynamics.obs_step(
        None,
        prev_action,
        embeds,
        proc['is_first'].squeeze(-1),
        sample=True
    )

    # 4) Decode
    dec_out = world_model.heads['decoder'](spat)
    recon   = dec_out['image'].mode()  # (B,H,W,C)

    # 5) Compute MSE loss against inputs
    target = proc['image'].squeeze(1)  # (B,H,W,C)
    loss   = (recon - target).pow(2).mean()

    # 6) Backward
    loss.backward()
    print("✅ Full world_model backward passed cleanly with batch size", batch_size)
    print(f"   loss = {loss.item():.4f}")

except RuntimeError as e:
    print("❌ RuntimeError during full-model backward:")
    print(e)
    import traceback; traceback.print_exc()

finally:
    torch.autograd.set_detect_anomaly(False)

#%% Cell 10: blockdiagGRU test
from networks import BlockDiagGRUCell

# Test parameters
batch = 4
T = 5
inp_size = 16
hidden_size = 32
blocks = 4

# Initialize the GRU cell
cell = BlockDiagGRUCell(inp_size, hidden_size, blocks, norm=True)

# Create initial hidden state and input sequence
h = torch.zeros(batch, hidden_size, requires_grad=True)
x_seq = torch.randn(T, batch, inp_size, requires_grad=True)

# Perform a multi-step "scan" through the GRU cell
h_list = []
h_t = h
for t in range(T):
    h_t, [h_t] = cell(x_seq[t], [h_t])
    h_list.append(h_t)

# Stack outputs and compute a dummy loss
H = torch.stack(h_list, dim=0)  # Shape: (T, batch, hidden_size)
loss = H.pow(2).mean()

# Backward pass with anomaly detection
torch.autograd.set_detect_anomaly(True)
loss.backward()
torch.autograd.set_detect_anomaly(False)

print("✅ Multi-step BPTT backward passed cleanly!")

cell = BlockDiagGRUCell(inp_size, hidden_size, blocks, norm=True)
h = torch.zeros(batch, hidden_size, requires_grad=True)
x_seq = torch.randn(T, batch, inp_size, requires_grad=True)

# a toy scan:
h_list = []
h_t = h
for t in range(T):
    h_t, [h_t] = cell(x_seq[t], [h_t])
    h_list.append(h_t)

# stack and a dummy loss over the whole trajectory:
H = torch.stack(h_list, 0)       # (T, B, hidden)
loss = H.pow(2).mean()
with torch.autograd.detect_anomaly():
    loss.backward()              # <- this will reproduce your training‐time error


#%% Cell 10: Summary
print("\n=== Summary ===")
print("If all tests passed, your hWorldModel is working correctly!")
print("You can now use it in the full training loop.")

# Clean up
env.close()

# %%
import torch
from networks import BlockDiagGRUCell

batch, inp, size, blocks = 4, 16, 32, 4
cell = BlockDiagGRUCell(inp, size, blocks, norm=True, act=torch.tanh)
h0 = torch.zeros(batch, size, requires_grad=True)
x  = torch.randn(batch, inp, requires_grad=True)

with torch.autograd.detect_anomaly():
    out, [h1] = cell(x, [h0])
    loss = out.pow(2).sum()
    loss.backward()
print("✅ minimal test passed")
# %%
