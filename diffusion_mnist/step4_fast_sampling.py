"""Step 4: sampling with fewer steps. Same network, same starting noise, fewer network calls."""

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tinymnist.model import UNet
from tinymnist.sampling import ddim_sample, ddpm_sample
from tinymnist.schedule import Schedule
from tinymnist.viz import show_grid

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="results")
args = parser.parse_args()

sched = Schedule()
model = UNet()
model.load_state_dict(torch.load(f"{args.out}/model.pt"))
model.eval()

runs = [(f"DDIM {s} steps", lambda s=s: ddim_sample(model, sched, 12, steps=s, guidance=0))
        for s in (3, 5, 10, 25, 50)]
runs.append((f"DDPM {sched.T} steps", lambda: ddpm_sample(model, sched, 12, guidance=0)))

fig, axes = plt.subplots(len(runs), 1, figsize=(8, 1.5 * len(runs)))
for ax, (name, fn) in zip(axes, runs):
    start = time.time()
    x = fn()
    secs = time.time() - start
    print(f"{name:16s} {secs:6.1f} s for 12 images")
    show_grid(ax, x, nrow=12)
    ax.set_ylabel(f"{name}\n{secs:.1f}s", rotation=0, ha="right", va="center", fontsize=8)
fig.suptitle("Same starting noise, different samplers")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{args.out}/4_fast_sampling.png", dpi=130)
print("Saved comparison to", args.out)
