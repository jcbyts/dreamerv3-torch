import torch
import numpy as np
import pytest
import gym
from unittest.mock import MagicMock

# Import the modules we're testing
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.cnvae import CNVAE
from networks.poisson import Poisson
import networks
from world_model_custom import WorldModelCustom


class TestConfig:
    """Mock config for testing."""
    def __init__(self, **kwargs):
        # Default values
        self.device = "cpu"
        self.precision = 32
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
        self.kl_free = 1.0
        self.dyn_scale = 0.5
        self.rep_scale = 0.1
        
        # Hierarchical parameters
        self.hierarchical_mode = False
        self.dyn_stoch_top = 16
        self.dyn_stoch_bottom = 16
        self.dyn_discrete_top = 32
        self.dyn_discrete_bottom = 32
        
        # New parameters
        self.use_cnvae = False
        self.use_poisson = False
        self.poisson_temp = 1.0
        self.cnvae_cfg = {
            'n_latent_scales': 4,
            'groups_per_scale': [2, 2, 1, 1],
            'latent_dim': 32,
            'image_channels': 3,
            'image_size': 64,
            'hidden_dim': 512,
            'act': 'SiLU',
            'norm': True
        }
        
        # Encoder/decoder configs
        self.encoder = {
            'mlp_keys': '$^', 'cnn_keys': 'image', 'act': 'SiLU', 'norm': True,
            'cnn_depth': 32, 'kernel_size': 4, 'minres': 4, 'mlp_layers': 5,
            'mlp_units': 1024, 'symlog_inputs': True
        }
        self.decoder = {
            'mlp_keys': '$^', 'cnn_keys': 'image', 'act': 'SiLU', 'norm': True,
            'cnn_depth': 32, 'kernel_size': 4, 'minres': 4, 'mlp_layers': 5,
            'mlp_units': 1024, 'cnn_sigmoid': False, 'image_dist': 'mse',
            'vector_dist': 'symlog_mse', 'outscale': 1.0
        }
        
        # Head configs
        self.reward_head = {'layers': 2, 'dist': 'symlog_mse', 'outscale': 1.0, 'loss_scale': 1.0}
        self.cont_head = {'layers': 2, 'outscale': 1.0, 'loss_scale': 1.0}
        self.grad_heads = ['decoder', 'reward', 'cont']
        self.units = 512
        
        # Optimizer configs
        self.model_lr = 3e-4
        self.opt_eps = 1e-5
        self.grad_clip = 100.0
        self.weight_decay = 1e-6
        self.opt = 'adam'
        
        # Override with any provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestPoissonDistribution:
    """Test the Poisson distribution implementation."""
    
    def test_poisson_creation(self):
        """Test basic Poisson distribution creation."""
        log_rate = torch.randn(10, 5)
        poisson = Poisson(log_rate, t=1.0)
        
        assert poisson.log_rate.shape == log_rate.shape
        assert poisson.rate.shape == log_rate.shape
        assert poisson.t == 1.0
        
    def test_poisson_sampling(self):
        """Test Poisson sampling methods."""
        log_rate = torch.ones(5, 3) * 0.5  # rate ≈ 1.65
        poisson = Poisson(log_rate, t=1.0)
        
        # Test reparameterized sampling
        samples = poisson.rsample()
        assert samples.shape == (5, 3)
        assert torch.all(samples >= 0)  # Poisson samples are non-negative
        
        # Test standard sampling
        samples_std = poisson.sample()
        assert samples_std.shape == (5, 3)
        assert torch.all(samples_std >= 0)
        
    def test_poisson_log_prob(self):
        """Test Poisson log probability computation."""
        log_rate = torch.ones(3, 2) * 0.0  # rate = 1.0
        poisson = Poisson(log_rate, t=1.0)
        
        values = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.float)
        log_probs = poisson.log_prob(values)
        
        assert log_probs.shape == (3, 2)
        # For rate=1, P(0) = e^(-1) ≈ 0.368, log_prob ≈ -1
        assert torch.allclose(log_probs[0, 0], torch.tensor(-1.0), atol=1e-3)
        
    def test_poisson_kl_divergence(self):
        """Test Poisson KL divergence computation."""
        log_rate1 = torch.ones(2, 3) * 0.5
        log_rate2 = torch.ones(2, 3) * 1.0
        
        poisson1 = Poisson(log_rate1, t=1.0)
        poisson2 = Poisson(log_rate2, t=1.0)
        
        kl = poisson1.kl(poisson2)
        assert kl.shape == (2, 3)
        assert torch.all(kl >= 0)  # KL divergence is non-negative


class TestCNVAE:
    """Test the CNVAE implementation."""
    
    def test_cnvae_creation(self):
        """Test CNVAE creation with default config."""
        cfg = {
            'n_latent_scales': 3,
            'groups_per_scale': [2, 2, 1],
            'latent_dim': 16,
            'image_channels': 3,
            'image_size': 64,
            'hidden_dim': 256,
            'act': 'SiLU',
            'norm': True
        }
        
        cnvae = CNVAE(cfg)
        assert cnvae.n_scales == 3
        assert cnvae.groups_per_scale == [2, 2, 1]
        assert cnvae.latent_dim == 16
        
    def test_cnvae_forward(self):
        """Test CNVAE forward pass."""
        cfg = {
            'n_latent_scales': 2,
            'groups_per_scale': [2, 1],
            'latent_dim': 8,
            'image_channels': 3,
            'image_size': 32,
            'hidden_dim': 128,
            'act': 'SiLU',
            'norm': True
        }
        
        cnvae = CNVAE(cfg)
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)
        
        # Test xtract_ftr
        latents, features_enc, features_dec, q_all, p_all = cnvae.xtract_ftr(x, full=True)
        
        assert len(latents) == 2  # n_latent_scales
        assert len(q_all) == 2
        assert len(p_all) == 2
        
        # Test generate
        recon, _ = cnvae.generate(latents)
        assert recon.shape == x.shape
        
        # Test KL loss
        kl_all, kl_diag = cnvae.loss_kl(q_all, p_all)
        assert len(kl_all) == 2
        for kl in kl_all:
            assert kl.shape[0] == batch_size


class TestRSSMPoisson:
    """Test RSSM with Poisson latents."""
    
    def test_rssm_poisson_creation(self):
        """Test RSSM creation with Poisson latents."""
        rssm = networks.RSSM(
            stoch=16,
            deter=128,
            hidden=128,
            discrete=False,
            use_poisson=True,
            poisson_temp=0.5,
            num_actions=4,
            embed=64,
            device="cpu"
        )
        
        assert rssm._use_poisson == True
        assert rssm._poisson_temp == 0.5
        
    def test_rssm_poisson_forward(self):
        """Test RSSM forward pass with Poisson latents."""
        rssm = networks.RSSM(
            stoch=8,
            deter=64,
            hidden=64,
            discrete=False,
            use_poisson=True,
            poisson_temp=1.0,
            num_actions=4,
            embed=32,
            device="cpu"
        )
        
        batch_size = 3
        seq_len = 5
        
        # Test initial state
        state = rssm.initial(batch_size)
        assert "log_rate" in state
        assert "stoch" in state
        assert "deter" in state
        
        # Test observe
        embed = torch.randn(batch_size, seq_len, 32)
        action = torch.randn(batch_size, seq_len, 4)
        is_first = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        post, prior = rssm.observe(embed, action, is_first)
        
        assert "log_rate" in post
        assert "log_rate" in prior
        assert post["log_rate"].shape == (batch_size, seq_len, 8)


class TestWorldModelCustom:
    """Test the custom WorldModel implementation."""
    
    def create_mock_spaces(self):
        """Create mock observation and action spaces."""
        obs_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        })
        act_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        return obs_space, act_space
        
    def test_vanilla_dreamer_compatibility(self):
        """Test that vanilla Dreamer still works."""
        obs_space, act_space = self.create_mock_spaces()
        config = TestConfig()
        
        wm = WorldModelCustom(obs_space, act_space, 0, config)
        
        # Should have regular encoder/decoder
        assert hasattr(wm, 'encoder')
        assert not hasattr(wm, 'bottleneck')
        assert 'decoder' in wm.heads
        
    def test_poisson_dreamer(self):
        """Test Dreamer with Poisson latents."""
        obs_space, act_space = self.create_mock_spaces()
        config = TestConfig(use_poisson=True, poisson_temp=0.5)
        
        wm = WorldModelCustom(obs_space, act_space, 0, config)
        
        # Should have regular encoder but Poisson RSSM
        assert hasattr(wm, 'encoder')
        assert wm.dynamics._use_poisson == True
        assert wm.dynamics._poisson_temp == 0.5
        
    def test_cnvae_hdreamer(self):
        """Test hDreamer with CNVAE."""
        obs_space, act_space = self.create_mock_spaces()
        config = TestConfig(
            use_cnvae=True,
            hierarchical_mode=True,
            dyn_stoch_top=8,
            dyn_stoch_bottom=8
        )
        
        wm = WorldModelCustom(obs_space, act_space, 0, config)
        
        # Should have CNVAE bottleneck
        assert hasattr(wm, 'bottleneck')
        assert not hasattr(wm, 'encoder')
        assert hasattr(wm, 'decoder')  # decoder should point to bottleneck
        
    def test_full_cnvae_poisson(self):
        """Test full CNVAE + Poisson configuration."""
        obs_space, act_space = self.create_mock_spaces()
        config = TestConfig(
            use_cnvae=True,
            use_poisson=True,
            poisson_temp=0.5,
            hierarchical_mode=True
        )
        
        wm = WorldModelCustom(obs_space, act_space, 0, config)
        
        # Should have both CNVAE and Poisson
        assert hasattr(wm, 'bottleneck')
        assert wm.dynamics._use_poisson == True
        
    def test_training_step(self):
        """Test training step with different configurations."""
        obs_space, act_space = self.create_mock_spaces()
        
        # Test data
        batch_size, seq_len = 2, 8
        data = {
            'image': torch.rand(batch_size, seq_len, 64, 64, 3),
            'action': torch.rand(batch_size, seq_len, 4),
            'reward': torch.rand(batch_size, seq_len),
            'is_first': torch.zeros(batch_size, seq_len, dtype=torch.bool),
            'is_terminal': torch.zeros(batch_size, seq_len, dtype=torch.bool),
        }
        
        # Test vanilla Dreamer
        config = TestConfig()
        wm = WorldModelCustom(obs_space, act_space, 0, config)
        post, context, metrics = wm._train(data)
        
        assert 'kl' in metrics
        assert 'image_loss' in metrics
        assert 'reward_loss' in metrics
        
        # Test Poisson Dreamer
        config_poisson = TestConfig(use_poisson=True)
        wm_poisson = WorldModelCustom(obs_space, act_space, 0, config_poisson)
        post_p, context_p, metrics_p = wm_poisson._train(data)
        
        assert 'kl' in metrics_p
        assert 'image_loss' in metrics_p


if __name__ == "__main__":
    # Run basic tests
    test_poisson = TestPoissonDistribution()
    test_poisson.test_poisson_creation()
    test_poisson.test_poisson_sampling()
    test_poisson.test_poisson_log_prob()
    test_poisson.test_poisson_kl_divergence()
    print("✓ Poisson distribution tests passed")
    
    test_cnvae = TestCNVAE()
    test_cnvae.test_cnvae_creation()
    test_cnvae.test_cnvae_forward()
    print("✓ CNVAE tests passed")
    
    test_rssm = TestRSSMPoisson()
    test_rssm.test_rssm_poisson_creation()
    test_rssm.test_rssm_poisson_forward()
    print("✓ RSSM Poisson tests passed")
    
    test_wm = TestWorldModelCustom()
    test_wm.test_vanilla_dreamer_compatibility()
    test_wm.test_poisson_dreamer()
    test_wm.test_cnvae_hdreamer()
    test_wm.test_full_cnvae_poisson()
    test_wm.test_training_step()
    print("✓ WorldModelCustom tests passed")
    
    print("All tests passed! ✓")
