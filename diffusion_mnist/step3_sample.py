"""Step 3: the reverse process. Start from pure noise and denoise step by step."""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
from torchvision.utils import make_grid

from tinymnist.data import to_display_range
from tinymnist.device import DEVICE
from tinymnist.model import load_model
from tinymnist.sampling import ddpm_sample
from tinymnist.schedule import Schedule
from tinymnist.viz import show_grid

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="results")
args = parser.parse_args()

sched = Schedule(device=DEVICE)
model = load_model(args.out, DEVICE)

# No label (guidance 0): the model invents digits on its own.
samples, snaps = ddpm_sample(model, sched, n=64, labels=None, guidance=0, keep_every=10)

fig, ax = plt.subplots(figsize=(6, 6))
show_grid(ax, samples, nrow=8)
ax.set_title("64 digits generated from pure noise (no label)")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{args.out}/3_generated_digits.png", dpi=130)

strip_t = [1000, 800, 600, 400, 200, 100, 0]
fig, axes = plt.subplots(1, len(strip_t), figsize=(2.0 * len(strip_t), 2.6))
for ax, t in zip(axes, strip_t):
    snap = snaps[(1000 - t) // 10]
    show_grid(ax, snap[:4], nrow=2)
    ax.set_title(f"t = {t}", fontsize=9)
fig.suptitle("Reverse process: noise to digits")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{args.out}/3_reverse_process.png", dpi=130)

fig, ax = plt.subplots(figsize=(3, 3))
frames = [s[:9] for s in snaps[::2]] + [snaps[-1][:9]]  # every 20th step, 3x3 digits
show_grid(ax, frames[0], nrow=3)
image = ax.images[0]
title = ax.set_title("")


def update(i):
    i = min(i, len(frames) - 1)
    image.set_data(make_grid(to_display_range(frames[i].cpu()), nrow=3, padding=1, pad_value=1.0)[0].numpy())
    title.set_text(f"Denoising: t = {max(sched.T - i * 20, 0)}")
    return image, title


anim = FuncAnimation(fig, update, frames=len(frames) + 8, interval=80)
anim.save(f"{args.out}/3_reverse_process.gif", writer=PillowWriter(fps=12))
print("Saved samples and animation to", args.out)
