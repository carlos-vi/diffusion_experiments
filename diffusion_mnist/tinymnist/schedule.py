"""The noise schedule and the forward (noising) process q(x_t | x_0) for images."""

import torch


class Schedule:
    """beta_t: noise added at step t. alpha_bar_t: fraction of the original signal left at step t."""

    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        prev = torch.cat([torch.ones(1), self.alpha_bars[:-1]])
        self.posterior_var = self.betas * (1.0 - prev) / (1.0 - self.alpha_bars)

    def q_sample(self, x0, t, noise=None):
        """x_t = sqrt(ab_t) * x_0 + sqrt(1 - ab_t) * eps, for a batch of images x0 of shape (B, 1, H, W)."""
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bars[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise, noise
