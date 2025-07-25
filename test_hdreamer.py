#!/usr/bin/env python3
"""
Unit tests for hDreamer hierarchical extension.
Tests hierarchical latent shapes, KL computation, state concatenation, and backward compatibility.
"""

import torch
import numpy as np
import sys
import os

# Add the dreamerv3-torch directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import networks
import models
import tools


class TestConfig:
    """Mock config for testing."""
    def __init__(self, hierarchical_mode=False):
        self.device = "cpu"
        self.dyn_stoch = 32
        self.dyn_deter = 512
        self.dyn_hidden = 512
        self.dyn_rec_depth = 1
        self.dyn_discrete = 32
        self.act = "SiLU"
        self.norm = True
        self.dyn_mean_act = "none"
        self.dyn_std_act = "sigmoid2"
        self.dyn_min_std = 0.1
        self.unimix_ratio = 0.01
        self.initial = "learned"
        self.num_actions = 4
        
        # Hierarchical parameters
        self.hierarchical_mode = hierarchical_mode
        self.dyn_stoch_top = 32
        self.dyn_stoch_bottom = 32
        self.dyn_discrete_top = 32
        self.dyn_discrete_bottom = 32

        # Additional config for WorldModel
        self.units = 512
        self.precision = 32
        self.model_lr = 3e-4
        self.opt_eps = 1e-5
        self.grad_clip = 100.0
        self.weight_decay = 1e-6
        self.opt = 'adam'
        self.grad_heads = ['decoder', 'reward', 'cont']


class TestHDreamerRSSM:
    """Test hierarchical RSSM implementation."""
    
    def test_flat_rssm_backward_compatibility(self):
        """Test that flat RSSM still works as before."""
        config = TestConfig(hierarchical_mode=False)
        embed_size = 1024
        
        rssm = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            embed_size,
            config.device,
        )
        
        batch_size = 16
        seq_len = 50
        
        # Test initial state
        initial_state = rssm.initial(batch_size)
        assert "stoch" in initial_state
        assert "deter" in initial_state
        assert "logit" in initial_state
        assert initial_state["stoch"].shape == (batch_size, config.dyn_stoch, config.dyn_discrete)
        
        # Test get_feat
        feat = rssm.get_feat(initial_state)
        expected_feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        assert feat.shape == (batch_size, expected_feat_size)
        
        print("✓ Flat RSSM backward compatibility test passed")
    
    def test_hierarchical_rssm_shapes(self):
        """Test hierarchical RSSM produces correct shapes."""
        config = TestConfig(hierarchical_mode=True)
        embed_size = 1024
        
        rssm = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            embed_size,
            config.device,
            hierarchical_mode=True,
            stoch_top=config.dyn_stoch_top,
            stoch_bottom=config.dyn_stoch_bottom,
            discrete_top=config.dyn_discrete_top,
            discrete_bottom=config.dyn_discrete_bottom,
        )
        
        batch_size = 16
        
        # Test initial state
        initial_state = rssm.initial(batch_size)
        assert "stoch_top" in initial_state
        assert "stoch_bottom" in initial_state
        assert "stoch" in initial_state  # for compatibility
        assert "deter" in initial_state
        assert "logit_top" in initial_state
        assert "logit_bottom" in initial_state
        
        assert initial_state["stoch_top"].shape == (batch_size, config.dyn_stoch_top, config.dyn_discrete_top)
        assert initial_state["stoch_bottom"].shape == (batch_size, config.dyn_stoch_bottom, config.dyn_discrete_bottom)
        
        # Test get_feat
        feat = rssm.get_feat(initial_state)
        expected_feat_size = (config.dyn_stoch_top * config.dyn_discrete_top + 
                             config.dyn_stoch_bottom * config.dyn_discrete_bottom + 
                             config.dyn_deter)
        assert feat.shape == (batch_size, expected_feat_size)
        
        print("✓ Hierarchical RSSM shapes test passed")
    
    def test_hierarchical_obs_step(self):
        """Test hierarchical observation step."""
        config = TestConfig(hierarchical_mode=True)
        embed_size = 1024
        
        rssm = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            embed_size,
            config.device,
            hierarchical_mode=True,
            stoch_top=config.dyn_stoch_top,
            stoch_bottom=config.dyn_stoch_bottom,
            discrete_top=config.dyn_discrete_top,
            discrete_bottom=config.dyn_discrete_bottom,
        )
        
        batch_size = 16
        prev_state = rssm.initial(batch_size)
        prev_action = torch.zeros(batch_size, config.num_actions)
        embed = torch.randn(batch_size, embed_size)
        is_first = torch.zeros(batch_size, 1)
        
        post, prior = rssm.obs_step(prev_state, prev_action, embed, is_first)
        
        # Check hierarchical structure
        assert "stoch_top" in post
        assert "stoch_bottom" in post
        assert "logit_top" in post
        assert "logit_bottom" in post
        assert "stoch" in post  # compatibility
        
        # Check shapes
        assert post["stoch_top"].shape == (batch_size, config.dyn_stoch_top, config.dyn_discrete_top)
        assert post["stoch_bottom"].shape == (batch_size, config.dyn_stoch_bottom, config.dyn_discrete_bottom)
        
        print("✓ Hierarchical obs_step test passed")
    
    def test_hierarchical_kl_loss(self):
        """Test hierarchical KL loss computation."""
        config = TestConfig(hierarchical_mode=True)
        embed_size = 1024
        
        rssm = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            embed_size,
            config.device,
            hierarchical_mode=True,
            stoch_top=config.dyn_stoch_top,
            stoch_bottom=config.dyn_stoch_bottom,
            discrete_top=config.dyn_discrete_top,
            discrete_bottom=config.dyn_discrete_bottom,
        )
        
        batch_size = 16
        prev_state = rssm.initial(batch_size)
        prev_action = torch.zeros(batch_size, config.num_actions)
        embed = torch.randn(batch_size, embed_size)
        is_first = torch.zeros(batch_size, 1)
        
        post, prior = rssm.obs_step(prev_state, prev_action, embed, is_first)
        
        # Test KL loss
        free = 1.0
        dyn_scale = 1.0
        rep_scale = 0.1
        
        loss, value, dyn_loss, rep_loss = rssm.kl_loss(post, prior, free, dyn_scale, rep_scale)
        
        # Check that losses are computed
        assert loss.shape == (batch_size,)
        assert value.shape == (batch_size,)
        assert dyn_loss.shape == (batch_size,)
        assert rep_loss.shape == (batch_size,)
        
        # Check that losses are reasonable (not NaN, not too large)
        assert torch.isfinite(loss).all()
        assert torch.isfinite(value).all()
        assert torch.isfinite(dyn_loss).all()
        assert torch.isfinite(rep_loss).all()
        
        print("✓ Hierarchical KL loss test passed")


def test_world_model_integration():
    """Test WorldModel integration with hierarchical RSSM."""
    import gym
    
    # Create mock observation and action spaces
    obs_space = gym.spaces.Dict({
        'image': gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
    })
    act_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    
    config = TestConfig(hierarchical_mode=True)
    config.encoder = {
        'mlp_keys': '$^', 'cnn_keys': 'image', 'act': 'SiLU', 'norm': True, 
        'cnn_depth': 32, 'kernel_size': 4, 'minres': 4, 'mlp_layers': 5, 
        'mlp_units': 1024, 'symlog_inputs': True
    }
    config.decoder = {
        'mlp_keys': '$^', 'cnn_keys': 'image', 'act': 'SiLU', 'norm': True, 
        'cnn_depth': 32, 'kernel_size': 4, 'minres': 4, 'mlp_layers': 5, 
        'mlp_units': 1024, 'cnn_sigmoid': False, 'image_dist': 'mse', 
        'vector_dist': 'symlog_mse', 'outscale': 1.0
    }
    config.reward_head = {'layers': 2, 'dist': 'symlog_mse', 'outscale': 1.0, 'loss_scale': 1.0}
    config.cont_head = {'layers': 2, 'outscale': 1.0, 'loss_scale': 1.0}
    config.grad_heads = ['decoder', 'reward', 'cont']
    config.model_lr = 3e-4
    config.opt_eps = 1e-5
    config.grad_clip = 100.0
    config.weight_decay = 1e-6
    config.opt = 'adam'
    config.precision = 32
    
    step = 0
    wm = models.WorldModel(obs_space, act_space, step, config)
    
    # Check that hierarchical parameters were passed correctly
    assert wm.dynamics._hierarchical_mode == True
    assert wm.dynamics._stoch_top == config.dyn_stoch_top
    assert wm.dynamics._stoch_bottom == config.dyn_stoch_bottom
    
    print("✓ WorldModel integration test passed")


if __name__ == "__main__":
    print("Running hDreamer unit tests...")
    
    test_rssm = TestHDreamerRSSM()
    test_rssm.test_flat_rssm_backward_compatibility()
    test_rssm.test_hierarchical_rssm_shapes()
    test_rssm.test_hierarchical_obs_step()
    test_rssm.test_hierarchical_kl_loss()
    
    test_world_model_integration()
    
    print("\n✅ All hDreamer tests passed!")
