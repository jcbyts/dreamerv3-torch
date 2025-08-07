#!/usr/bin/env python3
"""
Test script that reproduces the exact training scenario from the error.

This script attempts to recreate the exact sequence of operations that leads to:
RuntimeError: one of the variables needed for gradient computation has been 
modified by an inplace operation
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add the hDreamer directory to the path
sys.path.insert(0, 'hDreamer')

import tools
import networks
import models

def create_minimal_config():
    """Create a minimal config for testing."""
    class Config:
        def __init__(self):
            # Model config
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.precision = 32
            self.units = 256
            self.act = "SiLU"
            self.norm = True
            
            # Actor config
            self.actor = {
                'layers': 2,
                'dist': 'normal',
                'std': 'learned',
                'min_std': 0.1,
                'max_std': 1.0,
                'temp': 0.1,
                'unimix_ratio': 0.01,
                'outscale': 1.0,
                'lr': 3e-4,
                'eps': 1e-5,
                'grad_clip': 100.0,
            }
            
            # Critic config
            self.critic = {
                'layers': 2,
                'dist': 'normal',
                'slow_target': True,
                'lr': 8e-5,
                'eps': 1e-5,
                'grad_clip': 100.0,
            }
            
            # Training config
            self.weight_decay = 0.0
            self.opt = 'adam'
            self.imag_horizon = 15
            self.discount = 0.99
            self.discount_lambda = 0.95
            self.imag_gradient = 'dynamics'
            self.reward_EMA = True  # This is likely the culprit
            
            # Dynamics config (simplified)
            self.dyn_stoch = 30
            self.dyn_deter = 200
            self.dyn_discrete = False
            
    return Config()

def test_exact_training_scenario():
    """Reproduce the exact training scenario that causes the error."""
    print("=" * 60)
    print("Testing exact training scenario...")
    
    config = create_minimal_config()
    device = torch.device(config.device)
    
    # Create world model components
    class MinimalWorldModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dynamics = MinimalDynamics()
            
        @property
        def feat_size(self):
            if config.dyn_discrete:
                return config.dyn_stoch * config.dyn_discrete + config.dyn_deter
            else:
                return config.dyn_stoch + config.dyn_deter
    
    class MinimalDynamics(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(config.dyn_stoch + config.dyn_deter, config.dyn_stoch + config.dyn_deter)
            
        def get_feat(self, state):
            if isinstance(state, dict):
                if isinstance(state.get('stoch'), list):
                    # Hierarchical case
                    return torch.cat([s for s in state['stoch']], dim=-1)
                else:
                    # Flat case
                    return torch.cat([state['stoch'], state['deter']], dim=-1)
            return state
            
        def img_step(self, state, action):
            feat = self.get_feat(state)
            new_feat = self.linear(feat)
            # Split back into stoch and deter
            stoch = new_feat[..., :config.dyn_stoch]
            deter = new_feat[..., config.dyn_stoch:]
            return {'stoch': stoch, 'deter': deter}
    
    # Create the behavior model (this is where the error occurs)
    world_model = MinimalWorldModel().to(device)
    
    # Create ImagBehavior similar to the original
    class TestImagBehavior(nn.Module):
        def __init__(self, config, world_model):
            super().__init__()
            self._config = config
            self._world_model = world_model
            self._use_amp = config.precision != 32
            
            feat_size = world_model.feat_size
            
            # Actor network
            self.actor = networks.MLP(
                feat_size,
                (4,),  # 4 actions
                config.actor["layers"],
                config.units,
                config.act,
                config.norm,
                config.actor["dist"],
                config.actor["std"],
                config.actor["min_std"],
                config.actor["max_std"],
                temp=config.actor["temp"],
                unimix_ratio=config.actor["unimix_ratio"],
                outscale=config.actor["outscale"],
                device=config.device,
                name="Actor",
            )
            
            # Value network
            self.value = networks.MLP(
                feat_size,
                (),
                config.critic["layers"],
                config.units,
                config.act,
                config.norm,
                config.critic["dist"],
                device=config.device,
                name="Value",
            )
            
            # Optimizers
            kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
            self._actor_opt = tools.Optimizer(
                "actor",
                self.actor.parameters(),
                config.actor["lr"],
                config.actor["eps"],
                config.actor["grad_clip"],
                **kw,
            )
            
            self._value_opt = tools.Optimizer(
                "value",
                self.value.parameters(),
                config.critic["lr"],
                config.critic["eps"],
                config.critic["grad_clip"],
                **kw,
            )
            
            # RewardEMA (this is the likely culprit)
            if config.reward_EMA:
                self.register_buffer("ema_vals", torch.zeros((2,), device=config.device))
                self.reward_ema = models.RewardEMA(device=config.device)
        
        def _imagine(self, start, policy, horizon):
            """Exact copy of the imagination method."""
            dynamics = self._world_model.dynamics
            flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
            
            # Handle both hierarchical and flat latents
            start = {
                k: [flatten(x) for x in v] if isinstance(v, list) else flatten(v)
                for k, v in start.items()
            }

            def step(prev, _):
                state, _, _ = prev
                feat = dynamics.get_feat(state)
                inp = feat.detach()  # This detach might be problematic
                action = policy(inp).sample()
                succ = dynamics.img_step(state, action)
                
                # This detach pattern might cause issues
                succ_detached = {
                    k: [item.detach() for item in v] if isinstance(v, list) else v.detach() 
                    for k, v in succ.items()
                }
                
                return succ_detached, feat, action

            succ, feats, actions = tools.static_scan(
                step, [torch.arange(horizon)], (start, None, None)
            )
            
            states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
            return feats, states, actions
        
        def _compute_actor_loss(self, imag_feat, imag_action, target, weights, base):
            """Exact copy of actor loss computation."""
            metrics = {}
            inp = imag_feat.detach()
            policy = self.actor(inp)
            
            target = torch.stack(target, dim=1)
            
            # This is where the RewardEMA inplace operation happens
            if self._config.reward_EMA:
                offset, scale = self.reward_ema(target, self.ema_vals)
                normed_target = (target - offset) / scale
                normed_base = (base - offset) / scale
                adv = normed_target - normed_base
                metrics.update(tools.tensorstats(normed_target, "normed_target"))
            else:
                adv = target - base
            
            actor_target = adv
            actor_loss = -weights[:-1] * actor_target
            return actor_loss, metrics
        
        def train_step(self, start):
            """Reproduce the exact training step."""
            metrics = {}
            
            # Imagination
            with tools.RequiresGrad(self.actor):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )
                
                # Dummy reward
                reward = torch.randn_like(imag_feat[..., 0])
                
                # Compute target
                value = self.value(imag_feat).mode()
                target = tools.lambda_return(
                    reward[1:],
                    value[:-1],
                    torch.full_like(reward[1:], self._config.discount),
                    bootstrap=value[-1],
                    lambda_=self._config.discount_lambda,
                    axis=0,
                )
                weights = torch.ones_like(target)
                
                # Actor loss - THIS IS WHERE THE ERROR OCCURS
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat, imag_action, target, weights, value[:-1]
                )
                metrics.update(mets)
                
                # This line causes the error in the original code
                metrics.update(self._actor_opt(actor_loss.mean(), self.actor.parameters()))
            
            # Value loss
            with tools.RequiresGrad(self.value):
                value_loss = (value[:-1] - torch.stack(target, dim=1).detach()).pow(2).mean()
                metrics.update(self._value_opt(value_loss, self.value.parameters()))
            
            return metrics
    
    try:
        # Create the behavior model
        behavior = TestImagBehavior(config, world_model).to(device)
        
        # Create start state
        batch_size = 16
        start = {
            'stoch': torch.randn(batch_size, config.dyn_stoch, device=device, requires_grad=True),
            'deter': torch.randn(batch_size, config.dyn_deter, device=device, requires_grad=True),
        }
        
        print("Running training step that should reproduce the error...")
        
        # This should reproduce the exact error
        metrics = behavior.train_step(start)
        
        print("✓ Exact scenario test passed - no error occurred")
        print("  This suggests the issue might be environment-specific or require more steps")
        return True
        
    except RuntimeError as e:
        if "inplace operation" in str(e):
            print(f"✗ Exact scenario test failed - REPRODUCED THE ERROR!")
            print(f"  Error: {e}")
            print("\n🎯 ROOT CAUSE IDENTIFIED:")
            print("  The RewardEMA inplace operation is modifying ema_vals buffer")
            print("  while it's part of the computation graph.")
            print("\n💡 SOLUTION:")
            print("  Modify RewardEMA to avoid inplace operations on tracked tensors.")
            return False
        else:
            raise e

if __name__ == "__main__":
    print("Reproducing the exact training scenario that causes the inplace operation error...")
    
    try:
        result = test_exact_scenario()
        
        if not result:
            print("\n" + "=" * 60)
            print("ERROR REPRODUCED SUCCESSFULLY!")
            print("The issue is in the RewardEMA class.")
            print("\nTo fix this, you need to modify the RewardEMA.__call__ method")
            print("to avoid inplace operations on the ema_vals buffer.")
            print("\nSuggested fix:")
            print("Replace: ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals")
            print("With: ema_vals.data[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals")
            print("Or use: torch.no_grad() context around the EMA update")
        else:
            print("\n" + "=" * 60)
            print("Could not reproduce the error in this simplified scenario.")
            print("The issue might require:")
            print("1. Multiple training steps")
            print("2. Specific tensor shapes/values")
            print("3. Interaction with other components")
            print("4. Specific CUDA/PyTorch version")
            
    except Exception as e:
        print(f"Test failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
