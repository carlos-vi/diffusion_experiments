"""The noise schedule and the forward (noising) process q(x_t | x_0)."""

import torch


class Schedule:
    """Precomputes everything we need from the betas.

    beta_t       : variance of the noise added at step t
    alpha_t      : 1 - beta_t
    alpha_bar_t  : prod_{s<=t} alpha_s, the fraction of the original signal that survives to step t
    """

    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        prev = torch.cat([torch.ones(1), self.alpha_bars[:-1]])
        # variance of the true posterior q(x_{t-1} | x_t, x_0), used when sampling
        self.posterior_var = self.betas * (1.0 - prev) / (1.0 - self.alpha_bars)

    def q_sample(self, x0, t, noise=None):
        """Jump straight to step t:  x_t = sqrt(ab_t) * x_0 + sqrt(1 - ab_t) * eps."""
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bars[t].unsqueeze(-1)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise, noise
