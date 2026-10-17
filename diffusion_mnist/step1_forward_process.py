"""Step 1: the forward process on real digits. Watch them dissolve into noise."""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tinymnist.data import load_mnist, to_model_range
from tinymnist.schedule import Schedule
from tinymnist.viz import show_grid

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="results")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

torch.manual_seed(0)
sched = Schedule()
images, labels = load_mnist()
x0 = to_model_range(images[:6])

steps = [0, 50, 100, 200, 400, 700, 999]
rows = []
for t in steps:
    xt, _ = sched.q_sample(x0, torch.full((len(x0),), t))
    rows.append(xt)

fig, axes = plt.subplots(1, len(steps), figsize=(2.0 * len(steps), 3.4))
for ax, t, xt in zip(axes, steps, rows):
    show_grid(ax, xt, nrow=1)
    ax.set_title(f"t = {t}\nsignal kept: {sched.alpha_bars[t].sqrt():.2f}", fontsize=8)
fig.suptitle("Forward process on MNIST digits")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{args.out}/1_forward_process.png", dpi=130)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3))
a1.plot(sched.betas); a1.set_title("beta_t (noise added per step)"); a1.set_xlabel("t")
a2.plot(sched.alpha_bars); a2.set_title("alpha_bar_t (fraction of signal left)"); a2.set_xlabel("t")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{args.out}/1_noise_schedule.png", dpi=120)
print("Saved forward process figures to", args.out)
