import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latentatk.neural.cache_fusion import (
    RoleLayerResidualFusion, residual_fusion_loss,
)


def test_role_layer_residual_fusion_paths_and_gradients():
    torch.manual_seed(0)
    model = RoleLayerResidualFusion([12, 16], hidden=24, n_layers=3, device="cpu")
    small = torch.randn(5, 4, 3, 12)
    large = torch.randn(5, 4, 3, 16)
    both = model.forward(small, large)
    assert both["role_probs"].shape == (5, 4)
    assert both["topology_logits"].shape == (5, 4)
    assert both["bridge_weights"].shape == (5, 5)
    assert both["depth_logits"].shape == (5, 6)
    assert both["gates"].shape == (5, 4)
    assert torch.all((both["gates"] > 0) & (both["gates"] < 1))

    small_only = torch.tensor([[True, False]]).repeat(5, 1)
    large_only = torch.tensor([[False, True]]).repeat(5, 1)
    assert torch.equal(model.forward(small, large, small_only)["gates"], torch.ones(5, 4))
    assert torch.equal(model.forward(small, large, large_only)["gates"], torch.zeros(5, 4))

    targets = (
        torch.softmax(torch.randn(5, 4), -1), torch.randint(0, 4, (5,)),
        torch.randn(5, 5), torch.randint(0, 6, (5,)),
    )
    loss, _ = residual_fusion_loss(model, small, large, targets)
    loss.total.backward()
    assert torch.isfinite(loss.total)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_fusion_rejects_rows_with_no_expert():
    model = RoleLayerResidualFusion([4, 6], hidden=8, n_layers=2, device="cpu")
    small, large = torch.randn(1, 4, 2, 4), torch.randn(1, 4, 2, 6)
    try:
        model.forward(small, large, torch.zeros(1, 2, dtype=torch.bool))
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("missing experts should fail")
