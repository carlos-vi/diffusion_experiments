"""Plot helper: show a batch of images as a grid."""

import torch
from torchvision.utils import make_grid

from .data import to_display_range


def show_grid(ax, x, nrow):
    grid = make_grid(to_display_range(x.cpu()), nrow=nrow, padding=1, pad_value=1.0)
    ax.imshow(grid[0].numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
