"""Reverse processes: turn pure noise into data, one denoising step at a time."""

import torch


@torch.no_grad()
def ddpm_sample(model, sched, n, seed=0, keep_every=0):
    """Ancestral sampling (DDPM): T stochastic steps.

    If keep_every > 0, also returns a list of snapshots of x_t for animation.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g)
    snaps = [x.clone()] if keep_every else None
    for t in reversed(range(sched.T)):
        tt = torch.full((n,), t, dtype=torch.long)
        eps = model(x, tt)
        a, ab, b = sched.alphas[t], sched.alpha_bars[t], sched.betas[t]
        mean = (x - b / (1.0 - ab).sqrt() * eps) / a.sqrt()
        if t > 0:
            x = mean + sched.posterior_var[t].sqrt() * torch.randn(n, 2, generator=g)
        else:
            x = mean
        if keep_every and (t % keep_every == 0 or t == 0):
            snaps.append(x.clone())
    return (x, snaps) if keep_every else x


@torch.no_grad()
def ddim_sample(model, sched, n, steps=20, seed=0):
    """Deterministic DDIM sampling with only `steps` network evaluations."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g)
    ts = torch.linspace(sched.T - 1, 0, steps).round().long()
    for i, t in enumerate(ts):
        tt = torch.full((n,), int(t), dtype=torch.long)
        eps = model(x, tt)
        ab = sched.alpha_bars[t]
        x0 = (x - (1.0 - ab).sqrt() * eps) / ab.sqrt()  # current guess of the clean point
        ab_prev = sched.alpha_bars[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
        x = ab_prev.sqrt() * x0 + (1.0 - ab_prev).sqrt() * eps
    return x
