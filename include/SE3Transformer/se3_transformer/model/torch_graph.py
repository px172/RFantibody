import os
from typing import Optional, Tuple

import torch
from torch import Tensor


class TorchGraph:
    """Small edge-list graph with the DGL subset used by SE3Transformer."""

    is_torch_graph = True

    def __init__(self, edges: Tuple[Tensor, Tensor], num_nodes: int):
        src, dst = edges
        self._src = src.long()
        self._dst = dst.long()
        self._num_nodes = int(num_nodes)
        self.edata = {}

    def edges(self):
        return self._src, self._dst

    def num_nodes(self):
        return self._num_nodes

    def to(self, device=None):
        self._src = self._src.to(device)
        self._dst = self._dst.to(device)
        self.edata = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in self.edata.items()
        }
        return self


def is_torch_graph(graph) -> bool:
    return getattr(graph, "is_torch_graph", False)


def _dgl_available() -> bool:
    try:
        import dgl  # noqa: F401
    except Exception:
        return False
    return True


def get_graph_backend(device: Optional[torch.device] = None) -> str:
    backend = os.getenv("RFANTIBODY_SE3_BACKEND", "auto").lower()
    if backend in {"torch", "dgl"}:
        return backend
    if device is not None and torch.device(device).type == "xpu":
        return "torch"
    return "dgl" if _dgl_available() else "torch"


def make_graph(edges: Tuple[Tensor, Tensor], num_nodes: int, backend: Optional[str] = None):
    src, _dst = edges
    selected_backend = backend or get_graph_backend(src.device)
    if selected_backend == "torch":
        return TorchGraph(edges, num_nodes=num_nodes)
    try:
        import dgl
    except Exception as exc:
        raise ImportError(
            "DGL backend requested but DGL is not importable. "
            "Set RFANTIBODY_SE3_BACKEND=torch to use the PyTorch edge-list backend."
        ) from exc
    return dgl.graph(edges, num_nodes=num_nodes)


def _num_nodes(graph) -> int:
    value = graph.num_nodes
    return int(value() if callable(value) else value)


def _zeros_like_nodes(graph, edge_values: Tensor) -> Tensor:
    return torch.zeros(
        (_num_nodes(graph),) + tuple(edge_values.shape[1:]),
        dtype=edge_values.dtype,
        device=edge_values.device,
    )


def _expand_index(index: Tensor, values: Tensor) -> Tensor:
    shape = (index.shape[0],) + (1,) * (values.dim() - 1)
    return index.reshape(shape).expand_as(values)


def copy_e_sum(graph, edge_values: Tensor) -> Tensor:
    _src, dst = graph.edges()
    out = _zeros_like_nodes(graph, edge_values)
    try:
        return out.index_add(0, dst, edge_values)
    except RuntimeError:
        for node in range(_num_nodes(graph)):
            mask = dst == node
            if torch.any(mask):
                out[node] = edge_values[mask].sum(dim=0)
        return out


def copy_e_mean(graph, edge_values: Tensor) -> Tensor:
    _src, dst = graph.edges()
    summed = copy_e_sum(graph, edge_values)
    counts = torch.zeros(
        (_num_nodes(graph),), dtype=edge_values.dtype, device=edge_values.device
    )
    ones = torch.ones_like(dst, dtype=edge_values.dtype)
    try:
        counts = counts.index_add(0, dst, ones)
    except RuntimeError:
        for node in range(_num_nodes(graph)):
            counts[node] = (dst == node).sum().to(edge_values.dtype)
    view_shape = (-1,) + (1,) * (edge_values.dim() - 1)
    return summed / counts.clamp_min(1).reshape(view_shape)


def e_dot_v(graph, edge_values: Tensor, node_values: Tensor) -> Tensor:
    _src, dst = graph.edges()
    return (edge_values * node_values[dst]).sum(dim=-1, keepdim=True)


def edge_softmax(graph, edge_scores: Tensor) -> Tensor:
    _src, dst = graph.edges()
    num_nodes = _num_nodes(graph)

    try:
        max_scores = torch.full(
            (num_nodes,) + tuple(edge_scores.shape[1:]),
            torch.finfo(edge_scores.dtype).min,
            dtype=edge_scores.dtype,
            device=edge_scores.device,
        )
        max_scores = max_scores.scatter_reduce(
            0, _expand_index(dst, edge_scores), edge_scores, reduce="amax"
        )
        exp_scores = torch.exp(edge_scores - max_scores[dst])
        denom = torch.zeros_like(max_scores).index_add(0, dst, exp_scores)
        return exp_scores / denom[dst].clamp_min(torch.finfo(edge_scores.dtype).tiny)
    except RuntimeError:
        out = torch.empty_like(edge_scores)
        for node in range(num_nodes):
            mask = dst == node
            if torch.any(mask):
                out[mask] = torch.softmax(edge_scores[mask], dim=0)
        return out
