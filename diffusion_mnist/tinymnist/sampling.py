"""Reverse processes for images, with optional classifier-free guidance."""

import torch

from .model import NULL_LABEL


def predict_eps(model, x, t, labels, guidance):
    """Noise prediction. guidance = 0 ignores the label, 1 follows it, > 1 follows it harder.

    Classifier-free guidance:  eps = eps_uncond + w * (eps_cond - eps_uncond).
    """
    tt = torch.full((len(x),), int(t), dtype=torch.long)
    null = torch.full((len(x),), NULL_LABEL, dtype=torch.long)
    if labels is None or guidance == 0:
        return model(x, tt, null)
    if guidance == 1:
        return model(x, tt, labels)
    both = model(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([null, labels]))
    eps_uncond, eps_cond = both.chunk(2)
    return eps_uncond + guidance * (eps_cond - eps_uncond)


@torch.no_grad()
def ddpm_sample(model, sched, n, labels=None, guidance=1.0, seed=0, keep_every=0):
    """Ancestral sampling: T stochastic steps. Optionally keeps snapshots of x_t."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 28, 28, generator=g)
    snaps = [x.clone()] if keep_every else None
    for t in reversed(range(sched.T)):
        eps = predict_eps(model, x, t, labels, guidance)
        a, ab, b = sched.alphas[t], sched.alpha_bars[t], sched.betas[t]
        x = (x - b / (1.0 - ab).sqrt() * eps) / a.sqrt()
        if t > 0:
            x = x + sched.posterior_var[t].sqrt() * torch.randn(x.shape, generator=g)
        if keep_every and (t % keep_every == 0 or t == 0):
            snaps.append(x.clone())
    return (x, snaps) if keep_every else x


@torch.no_grad()
def ddim_sample(model, sched, n, steps=50, labels=None, guidance=1.0, seed=0):
    """Deterministic DDIM sampling with only `steps` network evaluations."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 28, 28, generator=g)
    ts = torch.linspace(sched.T - 1, 0, steps).round().long()
    for i, t in enumerate(ts):
        eps = predict_eps(model, x, int(t), labels, guidance)
        ab = sched.alpha_bars[t]
        x0 = ((x - (1.0 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)  # guess of the clean image
        eps = (x - ab.sqrt() * x0) / (1.0 - ab).sqrt()  # noise consistent with the clamped guess
        ab_prev = sched.alpha_bars[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
        x = ab_prev.sqrt() * x0 + (1.0 - ab_prev).sqrt() * eps
    return x
