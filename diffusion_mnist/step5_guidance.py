"""Step 5: ask for a specific digit, and see how guidance strength changes the result.

Classifier-free guidance: eps = eps_uncond + w * (eps_cond - eps_uncond).
w = 0 ignores the label, w = 1 follows it, w > 1 follows it harder. A small separate classifier
is trained once (about 20 seconds) to measure how often the generated digit matches the request.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tinymnist.classifier import get_classifier
from tinymnist.model import UNet
from tinymnist.sampling import ddim_sample
from tinymnist.schedule import Schedule
from tinymnist.viz import show_grid

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="results")
args = parser.parse_args()

sched = Schedule()
model = UNet()
model.load_state_dict(torch.load(f"{args.out}/model.pt"))
model.eval()
clf = get_classifier(f"{args.out}/classifier.pt")

weights = [0, 1, 3, 6]
per_digit = 6
labels = torch.arange(10).repeat_interleave(per_digit)  # 0,0,0,0,0,0,1,1,...

fig, axes = plt.subplots(1, len(weights), figsize=(3.6 * len(weights), 6.4))
for ax, w in zip(axes, weights):
    x = ddim_sample(model, sched, len(labels), steps=50, labels=labels, guidance=w)
    with torch.no_grad():
        acc = (clf(x.clamp(-1, 1)).argmax(1) == labels).float().mean().item()
    print(f"guidance w = {w}: {100 * acc:.0f}% of generated digits are classified as the requested digit")
    show_grid(ax, x, nrow=per_digit)
    ax.set_title(f"w = {w}\nmatches request: {100 * acc:.0f}%", fontsize=10)
fig.suptitle("Rows are digits 0 to 9, requested from the model")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{args.out}/5_guidance.png", dpi=130)
print("Saved guidance figure to", args.out)
