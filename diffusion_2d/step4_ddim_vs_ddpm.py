"""Step 4: sampling with fewer steps. DDIM needs far fewer network calls than DDPM."""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tinydiff.data import make_dataset
from tinydiff.metrics import coverage_error
from tinydiff.model import NoisePredictor
from tinydiff.sampling import ddim_sample, ddpm_sample
from tinydiff.schedule import Schedule

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="moons", choices=["moons", "swiss"])
parser.add_argument("--out", default="results")
args = parser.parse_args()

sched = Schedule()
model = NoisePredictor()
model.load_state_dict(torch.load(f"{args.out}/model_{args.dataset}.pt"))
model.eval()
real = make_dataset(args.dataset, n=2000, seed=1)

runs = [("DDIM, 2 steps", ddim_sample(model, sched, 2000, steps=2)),
        ("DDIM, 5 steps", ddim_sample(model, sched, 2000, steps=5)),
        ("DDIM, 20 steps", ddim_sample(model, sched, 2000, steps=20)),
        ("DDIM, 100 steps", ddim_sample(model, sched, 2000, steps=100)),
        (f"DDPM, {sched.T} steps", ddpm_sample(model, sched, 2000))]

fig, axes = plt.subplots(1, len(runs), figsize=(3 * len(runs), 3.2), sharex=True, sharey=True)
for ax, (name, x) in zip(axes, runs):
    err = coverage_error(x, real)
    print(f"{name:18s} distance to nearest real point: {err:.4f}")
    ax.scatter(x[:, 0], x[:, 1], s=2)
    ax.set_title(f"{name}\nerror {err:.3f}", fontsize=9)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(f"{args.out}/4_ddim_vs_ddpm_{args.dataset}.png", dpi=120)
print("Saved comparison to", args.out)
