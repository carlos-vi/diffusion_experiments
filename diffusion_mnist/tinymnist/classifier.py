"""A tiny digit classifier, used only to measure whether generated digits match the requested label."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import load_mnist, to_model_range


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


def get_classifier(path):
    """Loads the classifier from `path`, training it for one epoch first if it does not exist."""
    clf = Classifier()
    if os.path.exists(path):
        clf.load_state_dict(torch.load(path))
        return clf.eval()
    images, labels = load_mnist(train=True)
    test_images, test_labels = load_mnist(train=False)
    opt = torch.optim.Adam(clf.parameters(), lr=2e-3)
    torch.manual_seed(0)
    perm = torch.randperm(len(images))
    for i in range(0, len(images), 128):
        idx = perm[i:i + 128]
        loss = F.cross_entropy(clf(to_model_range(images[idx])), labels[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    clf.eval()
    with torch.no_grad():
        correct = sum((clf(to_model_range(test_images[i:i + 500])).argmax(1) == test_labels[i:i + 500]).sum().item()
                      for i in range(0, len(test_images), 500))
        acc = correct / len(test_images)
    print(f"Evaluation classifier trained: test accuracy {acc:.3f}")
    torch.save(clf.state_dict(), path)
    return clf
