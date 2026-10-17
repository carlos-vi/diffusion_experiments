"""A simple quality score for generated points."""

import torch


def coverage_error(generated, real):
    """Mean distance from each generated point to its nearest real point (lower is better)."""
    return torch.cdist(generated, real).min(dim=1).values.mean().item()
