"""Step 1: the forward process. Watch the data dissolve into Gaussian noise."""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tinydiff.data import make_dataset
from tinydiff.schedule import Schedule

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="moons", choices=["moons", "swiss"])
parser.add_argument("--out", default="results")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

torch.manual_seed(0)
sched = Schedule()
x0 = make_dataset(args.dataset, n=2000)

steps = [0, 10, 50, 100, 250, 500, 999]
fig, axes = plt.subplots(1, len(steps), figsize=(2.4 * len(steps), 2.6))
for ax, t in zip(axes, steps):
    xt, _ = sched.q_sample(x0, torch.full((len(x0),), t))
    ax.scatter(xt[:, 0], xt[:, 1], s=2, alpha=0.6)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
    ax.set_title(f"t = {t}\nsignal kept: {sched.alpha_bars[t].sqrt():.2f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Forward process: x_t = sqrt(ab_t) * x_0 + sqrt(1 - ab_t) * noise")
fig.tight_layout()
fig.savefig(f"{args.out}/1_forward_process_{args.dataset}.png", dpi=120)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3))
a1.plot(sched.betas); a1.set_title("beta_t (noise added per step)"); a1.set_xlabel("t")
a2.plot(sched.alpha_bars); a2.set_title("alpha_bar_t (fraction of signal left)"); a2.set_xlabel("t")
fig.tight_layout()
fig.savefig(f"{args.out}/1_noise_schedule.png", dpi=120)
print("Saved forward process figures to", args.out)
