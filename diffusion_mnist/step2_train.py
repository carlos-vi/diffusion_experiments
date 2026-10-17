"""Step 2: train the U-Net to predict the noise that was added to a digit.

The label is dropped 10% of the time so that the same network learns both "draw any digit"
and "draw this digit" (needed for classifier-free guidance in step 5).
"""

import argparse
import copy
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from tinymnist.data import load_mnist, to_model_range
from tinymnist.model import NULL_LABEL, UNet
from tinymnist.schedule import Schedule

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=5000, help="use e.g. 300 for a quick smoke test")
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--lr", type=float, default=2e-3)
parser.add_argument("--out", default="results")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

torch.manual_seed(0)
sched = Schedule()
images, labels = load_mnist()
model = UNet()
ema = copy.deepcopy(model)  # a slowly-moving average of the weights; it gives cleaner samples
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

losses = []
start = time.time()
for step in range(1, args.steps + 1):
    idx = torch.randint(0, len(images), (args.batch,))
    x0, y = to_model_range(images[idx]), labels[idx].clone()
    y[torch.rand(args.batch) < 0.1] = NULL_LABEL        # drop the label 10% of the time
    t = torch.randint(0, sched.T, (args.batch,))
    xt, noise = sched.q_sample(x0, t)
    loss = F.mse_loss(model(xt, t, y), noise)
    opt.zero_grad(); loss.backward(); opt.step(); lr_sched.step()
    with torch.no_grad():
        decay = min(0.995, (1 + step) / (10 + step))
        for pe, p in zip(ema.parameters(), model.parameters()):
            pe.mul_(decay).add_(p, alpha=1 - decay)
    losses.append(loss.item())
    if step % 200 == 0:
        elapsed = time.time() - start
        print(f"step {step:5d}  loss {sum(losses[-200:]) / 200:.4f}  ({elapsed:.0f}s elapsed)", flush=True)

torch.save(ema.state_dict(), f"{args.out}/model.pt")

k = min(50, max(1, len(losses) // 10))
smooth = torch.tensor(losses[: len(losses) // k * k]).view(-1, k).mean(1)
plt.figure(figsize=(5, 3))
plt.plot(torch.arange(len(smooth)) * k, smooth)
plt.xlabel("training step"); plt.ylabel("MSE (noise prediction)")
plt.title("Training loss"); plt.tight_layout()
plt.savefig(f"{args.out}/2_training_loss.png", dpi=120)
print("Saved model and loss curve to", args.out)
