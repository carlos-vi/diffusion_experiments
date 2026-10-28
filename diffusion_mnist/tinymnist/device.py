"""Picks the device once: the GPU if there is one, otherwise the CPU."""

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
