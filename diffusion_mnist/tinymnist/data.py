"""MNIST loading. Images are kept as uint8 (47 MB) and converted per batch to save RAM."""

import torch
from torchvision import datasets

DATA_DIR = "data"  # downloaded on first run (about 11 MB)


def load_mnist(train=True):
    ds = datasets.MNIST(DATA_DIR, train=train, download=True)
    return ds.data, ds.targets  # (N, 28, 28) uint8, (N,) int64


def to_model_range(images_uint8):
    """uint8 [0, 255] -> float [-1, 1] with a channel dimension: (B, 1, 28, 28)."""
    return (images_uint8.float() / 127.5 - 1.0).unsqueeze(1)


def to_display_range(x):
    """float [-1, 1] -> [0, 1] for plotting."""
    return ((x.clamp(-1, 1) + 1.0) / 2.0)
