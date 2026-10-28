"""Generate a digit of your choice with a trained model.

Example: python generate.py --digit 7 --out results_gpu
Uses DDIM (50 steps by default), so it takes a few seconds on CPU. The image goes to samples/.
"""

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tinymnist.device import DEVICE
from tinymnist.model import load_model
from tinymnist.sampling import ddim_sample
from tinymnist.schedule import Schedule
from tinymnist.viz import show_grid

parser = argparse.ArgumentParser()
parser.add_argument("--digit", type=int, required=True, choices=range(10), metavar="0-9")
parser.add_argument("--n", type=int, default=1, help="how many images of that digit")
parser.add_argument("--guidance", type=float, default=3.0, help="0 ignores the digit, 1 follows it, higher follows it harder")
parser.add_argument("--steps", type=int, default=50, help="DDIM steps; 10 is already recognisable")
parser.add_argument("--seed", type=int, default=0, help="change it to get a different image")
parser.add_argument("--out", default="results", help="folder that holds model.pt (e.g. results_gpu)")
args = parser.parse_args()

sched = Schedule(device=DEVICE)
model = load_model(args.out, DEVICE)
labels = torch.full((args.n,), args.digit, device=DEVICE)
x = ddim_sample(model, sched, args.n, steps=args.steps, labels=labels, guidance=args.guidance, seed=args.seed)

nrow = min(args.n, 8)
fig, ax = plt.subplots(figsize=(2 * nrow, 2 * math.ceil(args.n / nrow)))
show_grid(ax, x, nrow=nrow)
os.makedirs("samples", exist_ok=True)
path = f"samples/digit_{args.digit}_seed{args.seed}.png"
fig.savefig(path, dpi=130, bbox_inches="tight")
print("Saved", path)
