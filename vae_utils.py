import numpy as np
import torch
from typing import List, Tuple
from torch import nn

def beta_anneal_linear(
    n_iters: int,
    beta: float = 1.0,
    anneal_portion: float = 0.3,
    constant_portion: float = 0.0,
    min_beta: float = 1e-4,
) -> np.ndarray:
    """
    Linearly anneal β from min_beta → beta over anneal_portion of n_iters,
    with an initial constant portion at min_beta.
    """
    betas = np.ones(n_iters) * beta
    a = int(np.ceil(constant_portion * n_iters))
    b = int(np.ceil((constant_portion + anneal_portion) * n_iters))
    betas[:a] = min_beta
    betas[a:b] = np.linspace(min_beta, beta, b - a)
    return betas


def beta_anneal_cosine(
    n_iters: int,
    start: float = 0.0,
    stop: float = 1.0,
    n_cycles: int = 4,
    portion: float = 0.5,
    beta: float = 1.0,
) -> np.ndarray:
    """
    Cosine-anneal β over n_cycles, each cycle covering `portion` of its period.
    """
    period = n_iters / n_cycles
    step   = (stop - start) / (period * portion)
    betas  = np.ones(n_iters) * beta
    for c in range(n_cycles):
        v, i = start, 0
        while v <= stop and int(i + c*period) < n_iters:
            betas[int(i + c*period)] = (1 - np.cos(v * np.pi)) * beta / 2
            v += step
            i += 1
    return betas


def kl_balancer(
    kl_all: List[torch.Tensor],
    alpha: torch.Tensor = None,
    coeff: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given per-scale KLs (a list of [batch] tensors), stack them into [batch, scales],
    compute per-scale means, then either
      - weight & sum by gamma (if alpha is set and coeff < beta)
      - or simply sum across scales.
    Returns
      (balanced_kl [batch], gamma [scales], kl_vals [scales]).
    """
    kl_mat   = torch.stack(kl_all, dim=1)        # [B, L]
    kl_vals  = torch.mean(kl_mat, dim=0)         # [L]
    if alpha is not None and coeff < beta:
        # imbalance weights proportional to avg abs-KL × alpha
        gamma = torch.mean(kl_mat.detach().abs(), dim=0, keepdim=True)  # [1, L]
        gamma = gamma * alpha.unsqueeze(0)                              # [1, L]
        gamma = gamma / gamma.mean(dim=1, keepdim=True)                # normalize
        kl = (kl_mat * gamma).sum(dim=1)                               # [B]
        gamma = gamma.squeeze(0)                                       # [L]
    else:
        kl    = kl_mat.sum(dim=1)                                      # [B]
        gamma = torch.ones_like(kl_vals)                               # [L]
    return kl * coeff, gamma, kl_vals


def kl_balancer_coeff(
    groups: List[int],
    fun: str = 'equal',
) -> torch.Tensor:
    """
    Build a per-latent α vector from `groups` (number of latents per scale)
    and one of ['equal', 'linear', 'sqrt', 'square'] schemes.
    Normalizes so min(α)=1.
    """
    n = len(groups)
    if fun == 'equal':
        coeffs = torch.cat([torch.ones(groups[n - i - 1]) for i in range(n)], dim=0)
    elif fun == 'linear':
        coeffs = torch.cat([(2 ** i) * torch.ones(groups[n - i - 1]) for i in range(n)], dim=0)
    elif fun == 'sqrt':
        coeffs = torch.cat([(2 ** (i/2)) * torch.ones(groups[n - i - 1]) for i in range(n)], dim=0)
    elif fun == 'square':
        coeffs = torch.cat([((2 ** i) ** 2 / groups[n - i - 1]) * torch.ones(groups[n - i - 1]) for i in range(n)], dim=0)
    else:
        raise NotImplementedError(f"Unknown kl_balancer_coeff fun: {fun}")
    return coeffs / coeffs.min()


def add_weight_decay(
    model: nn.Module,
    weight_decay: float = 1e-2,
    skip: Tuple[str, ...] = ('bias',),
) -> list:
    """
    Split model.parameters() into two groups: 
      - no_decay: biases and 1-D params, with zero weight_decay 
      - decay: all other params
    Returns a list suitable for passing to torch.optim.AdamW(param_groups=...).
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) <= 1 or any(k in name for k in skip):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay,    'weight_decay': weight_decay},
    ]