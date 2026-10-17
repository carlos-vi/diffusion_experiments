"""A small MLP that predicts the noise eps that was added to x_0 to get x_t."""

import math

import torch
import torch.nn as nn


def time_embedding(t, dim=32):
    """Sinusoidal embedding of the integer step t, as in transformers."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    args = t.float().unsqueeze(-1) * freqs
    return torch.cat([args.sin(), args.cos()], dim=-1)


class NoisePredictor(nn.Module):
    def __init__(self, hidden=128, emb_dim=32):
        super().__init__()
        self.emb_dim = emb_dim
        self.net = nn.Sequential(
            nn.Linear(2 + emb_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, time_embedding(t, self.emb_dim)], dim=-1))
