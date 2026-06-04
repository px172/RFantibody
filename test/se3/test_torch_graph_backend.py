import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SE3_ROOT = PROJECT_ROOT / "include" / "SE3Transformer"
sys.path.insert(0, str(SE3_ROOT))

from se3_transformer.model import Fiber, SE3Transformer
from se3_transformer.model.torch_graph import (
    TorchGraph,
    copy_e_sum,
    e_dot_v,
    edge_softmax,
    make_graph,
)


def test_torch_graph_ops_match_manual_edge_reductions():
    src = torch.tensor([0, 0, 1, 2])
    dst = torch.tensor([1, 1, 2, 2])
    graph = TorchGraph((src, dst), num_nodes=3)

    edge_scores = torch.tensor([[1.0, 0.0], [2.0, 1.0], [0.0, 2.0], [3.0, 4.0]])
    softmax = edge_softmax(graph, edge_scores)

    assert torch.allclose(softmax[dst == 1].sum(dim=0), torch.ones(2))
    assert torch.allclose(softmax[dst == 2].sum(dim=0), torch.ones(2))

    edge_values = torch.arange(16, dtype=torch.float32).reshape(4, 2, 2)
    summed = copy_e_sum(graph, edge_values)

    expected = torch.zeros(3, 2, 2)
    expected[1] = edge_values[0] + edge_values[1]
    expected[2] = edge_values[2] + edge_values[3]
    assert torch.allclose(summed, expected)

    node_values = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    dots = e_dot_v(graph, edge_values, node_values)
    assert torch.allclose(dots[0], (edge_values[0] * node_values[1]).sum(dim=-1, keepdim=True))


def _small_se3_model():
    torch.manual_seed(0)
    return SE3Transformer(
        num_layers=1,
        fiber_in=Fiber({0: 2, 1: 1}),
        fiber_hidden=Fiber.create(2, 4),
        fiber_out=Fiber({0: 2, 1: 1}),
        num_heads=1,
        channels_div=1,
        fiber_edge=Fiber({0: 3}),
        use_layer_norm=True,
    ).eval()


def _small_inputs():
    torch.manual_seed(1)
    src = torch.tensor([0, 1, 2, 0, 3, 2])
    dst = torch.tensor([1, 2, 3, 3, 0, 0])
    rel_pos = torch.randn(src.shape[0], 3)
    node_feats = {
        "0": torch.randn(4, 2, 1),
        "1": torch.randn(4, 1, 3),
    }
    edge_feats = {
        "0": torch.randn(src.shape[0], 3, 1),
    }
    return src, dst, rel_pos, node_feats, edge_feats


def test_se3_transformer_runs_with_torch_graph_backend(monkeypatch):
    monkeypatch.setenv("RFANTIBODY_SE3_BACKEND", "torch")
    model = _small_se3_model()
    src, dst, rel_pos, node_feats, edge_feats = _small_inputs()

    graph = make_graph((src, dst), num_nodes=4)
    graph.edata["rel_pos"] = rel_pos
    out = model(graph, node_feats, edge_feats)

    assert set(out.keys()) == {"0", "1"}
    assert out["0"].shape == (4, 2, 1)
    assert out["1"].shape == (4, 1, 3)
    assert torch.isfinite(out["0"]).all()
    assert torch.isfinite(out["1"]).all()


def test_se3_torch_backend_matches_dgl_when_available(monkeypatch):
    dgl = pytest.importorskip("dgl")

    model = _small_se3_model()
    src, dst, rel_pos, node_feats, edge_feats = _small_inputs()

    dgl_graph = dgl.graph((src, dst), num_nodes=4)
    dgl_graph.edata["rel_pos"] = rel_pos
    expected = model(dgl_graph, node_feats, edge_feats)

    monkeypatch.setenv("RFANTIBODY_SE3_BACKEND", "torch")
    torch_graph = make_graph((src, dst), num_nodes=4)
    torch_graph.edata["rel_pos"] = rel_pos
    actual = model(torch_graph, node_feats, edge_feats)

    assert torch.allclose(actual["0"], expected["0"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual["1"], expected["1"], atol=1e-5, rtol=1e-5)
