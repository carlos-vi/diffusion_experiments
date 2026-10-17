"""2D toy datasets. Each returns an (n, 2) float tensor, normalised to zero mean / unit std."""

import numpy as np
import torch


def two_moons(n, noise=0.06, seed=0):
    rng = np.random.default_rng(seed)
    n_up = n // 2
    n_down = n - n_up
    a_up = rng.uniform(0, np.pi, n_up)
    a_down = rng.uniform(0, np.pi, n_down)
    up = np.stack([np.cos(a_up), np.sin(a_up)], axis=1)
    down = np.stack([1 - np.cos(a_down), 0.5 - np.sin(a_down)], axis=1)
    x = np.concatenate([up, down]) + rng.normal(0, noise, (n, 2))
    return x


def swiss_roll(n, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.uniform(0, 1, n))
    x = np.stack([t * np.cos(t), t * np.sin(t)], axis=1) / 3.0
    return x + rng.normal(0, noise, (n, 2))


DATASETS = {"moons": two_moons, "swiss": swiss_roll}


def make_dataset(name, n=10000, seed=0):
    x = DATASETS[name](n, seed=seed)
    x = (x - x.mean(0)) / x.std(0)
    return torch.tensor(x, dtype=torch.float32)
