"""Step 3: the reverse process. Start from pure noise and denoise step by step."""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch

from tinydiff.data import make_dataset
from tinydiff.metrics import coverage_error
from tinydiff.model import NoisePredictor
from tinydiff.sampling import ddpm_sample
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

samples, snaps = ddpm_sample(model, sched, n=2000, keep_every=10)
print(f"DDPM ({sched.T} steps) distance to nearest real point: {coverage_error(samples, real):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(7, 3.4), sharex=True, sharey=True)
axes[0].scatter(real[:, 0], real[:, 1], s=2, c="tab:green"); axes[0].set_title("Real data")
axes[1].scatter(samples[:, 0], samples[:, 1], s=2, c="tab:orange"); axes[1].set_title("Generated from noise")
for ax in axes:
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
fig.tight_layout()
fig.savefig(f"{args.out}/3_real_vs_generated_{args.dataset}.png", dpi=120)

fig, ax = plt.subplots(figsize=(4, 4))
sc = ax.scatter(snaps[0][:, 0], snaps[0][:, 1], s=2, c="tab:orange")
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); ax.set_xticks([]); ax.set_yticks([])
title = ax.set_title("")
n_frames = len(snaps)


def update(i):
    sc.set_offsets(snaps[min(i, n_frames - 1)].numpy())
    t = max(sched.T - i * 10, 0)
    title.set_text(f"Denoising: t = {t}")
    return sc, title


anim = FuncAnimation(fig, update, frames=n_frames + 10, interval=60)  # +10 frames to hold the end
anim.save(f"{args.out}/3_reverse_process_{args.dataset}.gif", writer=PillowWriter(fps=15))
print("Saved samples and animation to", args.out)
