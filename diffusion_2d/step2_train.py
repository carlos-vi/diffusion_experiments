"""Step 2: train the network to predict the noise that was added to a clean point."""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from tinydiff.data import make_dataset
from tinydiff.model import NoisePredictor
from tinydiff.schedule import Schedule

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="moons", choices=["moons", "swiss"])
parser.add_argument("--steps", type=int, default=8000)
parser.add_argument("--batch", type=int, default=512)
parser.add_argument("--lr", type=float, default=2e-3)
parser.add_argument("--out", default="results")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

torch.manual_seed(0)
sched = Schedule()
data = make_dataset(args.dataset, n=10000)
model = NoisePredictor()
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

losses = []
for step in range(1, args.steps + 1):
    x0 = data[torch.randint(0, len(data), (args.batch,))]
    t = torch.randint(0, sched.T, (args.batch,))       # a random noise level per sample
    xt, noise = sched.q_sample(x0, t)                   # noise the data...
    loss = F.mse_loss(model(xt, t), noise)              # ...and ask the model to find the noise
    opt.zero_grad(); loss.backward(); opt.step(); lr_sched.step()
    losses.append(loss.item())
    if step % 1000 == 0:
        print(f"step {step:5d}  loss {sum(losses[-200:]) / 200:.4f}")

torch.save(model.state_dict(), f"{args.out}/model_{args.dataset}.pt")

smooth = torch.tensor(losses).unfold(0, 100, 100).mean(1)
plt.figure(figsize=(5, 3))
plt.plot(torch.arange(len(smooth)) * 100, smooth)
plt.xlabel("training step"); plt.ylabel("MSE (noise prediction)")
plt.title("Training loss"); plt.tight_layout()
plt.savefig(f"{args.out}/2_training_loss_{args.dataset}.png", dpi=120)
print("Saved model and loss curve to", args.out)
