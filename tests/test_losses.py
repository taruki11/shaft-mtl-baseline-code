"""Tests for loss functions."""
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import masked_scaled_huber_loss


def test_masked_scaled_huber_basic():
    """Masked scaled huber loss is zero when pred == target."""
    pred = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    target = pred.clone()
    y_multi = torch.tensor([[1.0, 1.0, 1.0]])
    scale = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    loss = masked_scaled_huber_loss(pred, target, y_multi, scale=scale, beta=1.0)
    assert abs(loss.item()) < 1e-6


def test_masked_scaled_huber_respects_mask():
    """Loss only considers dimensions where the corresponding label is active."""
    pred = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    # r0 has y0=1 so it's active, r1-r6 have y0=0 so masked out
    target = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    y_multi = torch.tensor([[1.0, 0.0, 0.0]])
    scale = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # r0: pred=1.0, target=0.0, diff=1.0, scale=1.0, beta=1.0 -> 0.5*1.0*1.0/1.0 = 0.5
    # All other idx are masked, so only r0 contributes. Mean = 0.5/1 = 0.5
    loss = masked_scaled_huber_loss(pred, target, y_multi, scale=scale, beta=1.0)
    assert abs(loss.item() - 0.5) < 1e-5


def test_masked_scaled_huber_scale_effect():
    """Scaling reduces the loss magnitude."""
    pred = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    y_multi = torch.tensor([[1.0, 0.0, 0.0]])
    scale_small = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    scale_large = torch.tensor([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    loss_small = masked_scaled_huber_loss(pred, target, y_multi, scale=scale_small, beta=1.0)
    loss_large = masked_scaled_huber_loss(pred, target, y_multi, scale=scale_large, beta=1.0)
    # Larger scale => smaller normalized diff => smaller loss
    assert loss_large < loss_small


if __name__ == "__main__":
    test_masked_scaled_huber_basic()
    test_masked_scaled_huber_respects_mask()
    test_masked_scaled_huber_scale_effect()
    print("All loss tests passed.")
