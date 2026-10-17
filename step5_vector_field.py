"""Step 5: what did the model learn? Plot the direction it pushes points at different noise levels."""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tinydiff.data import make_dataset
from tinydiff.model import NoisePredictor
from tinydiff.schedule import Schedule

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="moons", choices=["moons", "swiss"])
parser.add_argument("--out", default="results")
args = parser.parse_args()

sched = Schedule()
model = NoisePredictor()
model.load_state_dict(torch.load(f"{args.out}/model_{args.dataset}.pt"))
model.eval()
real = make_dataset(args.dataset, n=1500, seed=1)

lin = torch.linspace(-3, 3, 24)
gx, gy = torch.meshgrid(lin, lin, indexing="xy")
grid = torch.stack([gx.flatten(), gy.flatten()], dim=1)

levels = [900, 500, 200, 50]
fig, axes = plt.subplots(1, len(levels), figsize=(3.4 * len(levels), 3.6))
for ax, t in zip(axes, levels):
    with torch.no_grad():
        eps = model(grid, torch.full((len(grid),), t))
    # the denoising direction is minus the predicted noise (this is the score, up to a scale)
    direction = -eps / (1.0 - sched.alpha_bars[t]).sqrt()
    ax.scatter(real[:, 0], real[:, 1], s=1, c="lightgray")
    ax.quiver(grid[:, 0], grid[:, 1], direction[:, 0], direction[:, 1], angles="xy", color="tab:red")
    ax.set_title(f"t = {t}"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Learned denoising direction (arrows) over the real data (gray)")
fig.tight_layout()
fig.savefig(f"{args.out}/5_vector_field_{args.dataset}.png", dpi=120)
print("Saved vector field to", args.out)
