"""A small U-Net (about 100k parameters) that predicts the noise in a 28x28 image.

It is conditioned on the step t and on the digit label. Label 10 means "no label": the model
is trained with the label randomly dropped, so one network can generate with or without it.
"""

import math

import torch
import torch.nn as nn

NULL_LABEL = 10


def time_embedding(t, dim=32):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    args = t.float().unsqueeze(-1) * freqs
    return torch.cat([args.sin(), args.cos()], dim=-1)


class Block(nn.Module):
    """conv -> norm -> act, add the (time + label) embedding, conv -> norm -> act, plus a skip."""

    def __init__(self, c_in, c_out, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, c_out)
        self.emb = nn.Linear(emb_dim, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.emb(emb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, base=16, emb_dim=64):
        super().__init__()
        self.t_mlp = nn.Sequential(nn.Linear(32, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.label_emb = nn.Embedding(11, emb_dim)  # digits 0-9 plus the "no label" token
        self.b1 = Block(1, base, emb_dim)                  # 28x28
        self.down1 = nn.Conv2d(base, base, 3, stride=2, padding=1)
        self.b2 = Block(base, 2 * base, emb_dim)           # 14x14
        self.down2 = nn.Conv2d(2 * base, 2 * base, 3, stride=2, padding=1)
        self.b3 = Block(2 * base, 2 * base, emb_dim)       # 7x7 (the bottleneck)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.b4 = Block(4 * base, 2 * base, emb_dim)       # 14x14, with the skip from b2
        self.b5 = Block(3 * base, base, emb_dim)           # 28x28, with the skip from b1
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x, t, y):
        emb = self.t_mlp(time_embedding(t)) + self.label_emb(y)
        h1 = self.b1(x, emb)
        h2 = self.b2(self.down1(h1), emb)
        h3 = self.b3(self.down2(h2), emb)
        u = self.b4(torch.cat([self.up(h3), h2], dim=1), emb)
        u = self.b5(torch.cat([self.up(u), h1], dim=1), emb)
        return self.out(u)
