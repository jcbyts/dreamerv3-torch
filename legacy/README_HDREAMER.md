# hDreamer: Hierarchical DreamerV3

This repository implements **hDreamer**, a hierarchical extension to DreamerV3 that adds two-level categorical latents while maintaining full backward compatibility.

## 🎯 Quick Start

### 1. Test the Implementation
```bash
# Verify core functionality works
python simple_test.py
```

### 2. Run a Quick Comparison
```bash
# Compare tiny models (2-3 hours on GPU)
python run_comparison.py --steps 50000 --seeds 3

# Full comparison (overnight)
python run_comparison.py --steps 200000 --seeds 5
```

### 3. Train Individual Models
```bash
# Train tiny DreamerV3
python dreamer.py --configs tiny_dreamerv3 --steps 100000 --logdir logs/dreamerv3

# Train tiny hDreamer  
python dreamer.py --configs tiny_hdreamer --steps 100000 --logdir logs/hdreamer
```

## 🔬 Model Configurations

### Matched Tiny Models (Fast Iteration)
| Model | Latent Structure | Total Capacity | Feature Size | Training Time |
|-------|------------------|----------------|--------------|---------------|
| **tiny_dreamerv3** | 12 × 32 flat | 384 | 640 | ~2 hours |
| **tiny_hdreamer** | 4×32 + 8×32 hierarchical | 384 | 640 | ~2 hours |

**Key Properties:**
- ✅ Same total latent capacity (384)
- ✅ Same feature dimensions (640) for fair comparison  
- ✅ Reduced model size for fast iteration
- ✅ Fits on 24GB GPU with 32 environments

### Standard Models
| Model | Configuration | Description |
|-------|---------------|-------------|
| **hdreamer** | 16×32 + 16×32 hierarchical | Full-size hierarchical model |
| **defaults** | 32×32 flat | Standard DreamerV3 |

## 🧠 How hDreamer Works

### Hierarchical Structure
- **Top Level (Coarse)**: Captures high-level patterns like ego motion
- **Bottom Level (Fine)**: Captures detailed dynamics like object interactions  
- **Residual Connection**: Bottom posterior = Bottom prior + learned delta

### Key Features
- **Two-level categorical latents** with residual posterior parameterization
- **Matched total capacity** ensures fair comparison with flat models
- **Backward compatibility** - can load existing DreamerV3 checkpoints
- **Same hyperparameters** - no additional tuning required

## 📊 Expected Results

### Primary Hypothesis
**hDreamer should show emergent ego/world factorization without auxiliary losses**

### Performance Expectations
- **Sample Efficiency**: 10-20% faster convergence
- **Task Performance**: Within ±5% of DreamerV3 (not worse)
- **Reconstruction**: Comparable or slightly better quality
- **Interpretability**: Clear hierarchical structure in latents

## 🔧 Implementation Details

### Core Changes
1. **RSSM Extension**: Added hierarchical latent support in `networks.py`
2. **Configuration**: Added hierarchical parameters to `configs.yaml`
3. **Backward Compatibility**: Safe checkpoint loading for existing models
4. **Feature Extraction**: Updated to handle [h, z_top, z_bottom] concatenation

### Monitoring
- **Per-level KL divergence** (target: 3-5 nats each)
- **Gradient norms** (clipping should trigger <1% of steps)
- **Hierarchical structure** via linear probing and MI analysis

## 📁 Repository Structure

```
dreamerv3-torch/
├── dreamer.py              # Main training script (unchanged)
├── networks.py             # RSSM with hierarchical support  
├── models.py               # WorldModel integration
├── configs.yaml            # Model configurations
├── simple_test.py          # Core functionality test
├── run_comparison.py       # DreamerV3 vs hDreamer comparison
├── legacy/                 # Moved complex experimental framework
└── results/                # Experiment outputs
```

## 🚀 Usage Examples

### Basic Training
```bash
# Standard DreamerV3
python dreamer.py --configs defaults --task vizdoom_basic

# hDreamer hierarchical
python dreamer.py --configs hdreamer --task vizdoom_basic
```

### Systematic Comparison
```bash
# Quick test (3 seeds, 50k steps)
python run_comparison.py --steps 50000 --seeds 3 --task vizdoom_basic

# Full comparison (5 seeds, 200k steps)  
python run_comparison.py --steps 200000 --seeds 5 --task vizdoom_basic
```

### Loading Existing Models
```python
# Load DreamerV3 checkpoint into hDreamer (automatic conversion)
python dreamer.py --configs hdreamer --task vizdoom_basic --resume path/to/dreamerv3/checkpoint
```

## ✅ Verification

### Core Tests
```bash
python simple_test.py
```
Should output:
```
🎉 All tests passed! hDreamer implementation is working correctly.
Ready for training experiments!
```

### Quick Comparison
```bash
python run_comparison.py --steps 10000 --seeds 2
```
Should complete without errors and show performance comparison.

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're in the `dreamerv3-torch` directory
2. **CUDA Errors**: Use `--device cpu` for CPU-only testing
3. **Config Errors**: Verify configs exist with `python -c "import yaml; print(list(yaml.safe_load(open('configs.yaml')).keys()))"`

### Debug Mode
```bash
# Test core functionality
python simple_test.py

# Check available configs
python -c "import yaml; configs = yaml.safe_load(open('configs.yaml')); print('Available configs:', list(configs.keys()))"
```

## 📚 Key Papers & References

- **DreamerV3**: Mastering Diverse Domains through World Models (Hafner et al., 2023)
- **Hierarchical VAEs**: Ladder Variational Autoencoders (Sønderby et al., 2016)
- **NVAE**: A Deep Hierarchical Variational Autoencoder (Vahdat & Kautz, 2020)

## 🎯 Next Steps

1. **Validate Implementation**: Run `python simple_test.py`
2. **Quick Experiment**: Run `python run_comparison.py --steps 50000 --seeds 3`
3. **Full Comparison**: Run overnight experiments with more seeds/steps
4. **Analysis**: Implement linear probing and MI analysis for factor disentanglement
5. **Scale Up**: Test on larger models and more complex environments

The implementation is ready for systematic experimentation! 🚀
