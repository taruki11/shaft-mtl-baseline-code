"""Tests for TCNProbeNet model."""
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_tcn_probe import TCNProbeNet


def test_tcn_probe_forward_basic():
    """TCNProbeNet forward pass produces correct output shapes."""
    model = TCNProbeNet(num_labels=3, reg_dim=7, in_ch=10, base_channels=16, tcn_levels=2)
    x = torch.randn(4, 10, 128)
    rpm = torch.tensor([600.0, 900.0, 1200.0, 1500.0])
    out = model(x, rpm=rpm)
    assert out["logits"].shape == (4, 3)
    assert out["reg"].shape == (4, 7)
    assert "feats" in out
    assert out["feats"]["fused"].shape == (4, 64)  # head_hidden=64


def test_tcn_probe_forward_no_rpm():
    """TCNProbeNet works with rpm=None (defaults to zeros)."""
    model = TCNProbeNet(num_labels=3, reg_dim=7, in_ch=10, base_channels=16, tcn_levels=2)
    x = torch.randn(2, 10, 64)
    out = model(x)
    assert out["logits"].shape == (2, 3)
    assert out["reg"].shape == (2, 7)


def test_tcn_probe_feats_keys():
    """TCNProbeNet feats dict contains expected keys (no legacy aliases)."""
    model = TCNProbeNet(num_labels=3, reg_dim=7, in_ch=10, base_channels=16, tcn_levels=2)
    x = torch.randn(2, 10, 64)
    out = model(x)
    feats = out["feats"]
    expected_keys = {"h_shared", "fault_concat", "fused", "speed_emb", "diag_feat", "order_feat", "orbit_img"}
    assert expected_keys.issubset(set(feats.keys()))
    # Legacy redundant keys should be removed
    legacy_keys = {"fa", "fo", "fc", "ea", "eo", "ec", "oa", "oo", "oc"}
    assert legacy_keys.isdisjoint(set(feats.keys()))


def test_tcn_probe_diag_feat_zero_init():
    """Diagnostic features are zero-initialized so they don't affect early training."""
    model = TCNProbeNet(num_labels=3, reg_dim=7, in_ch=10, base_channels=16, tcn_levels=2,
                        diag_feat_enable=1, diag_feat_hidden=32, diag_feat_zero_init=1)
    # The fusion_head input weight for diag_feat columns should be zero
    offset = 16 * 2 + 16  # h_shared + speed_feat
    assert torch.allclose(model.fusion_head[0].weight[:, offset:offset + 32], torch.zeros_like(model.fusion_head[0].weight[:, offset:offset + 32]))


def test_tcn_probe_gradient_flow():
    """Gradients flow through the model during backward pass."""
    model = TCNProbeNet(num_labels=3, reg_dim=7, in_ch=10, base_channels=16, tcn_levels=2)
    x = torch.randn(2, 10, 64, requires_grad=True)
    rpm = torch.tensor([600.0, 900.0])
    out = model(x, rpm=rpm)
    loss = out["logits"].sum() + out["reg"].sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


if __name__ == "__main__":
    test_tcn_probe_forward_basic()
    test_tcn_probe_forward_no_rpm()
    test_tcn_probe_feats_keys()
    test_tcn_probe_diag_feat_zero_init()
    test_tcn_probe_gradient_flow()
    print("All model tests passed.")
