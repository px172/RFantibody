# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Backend-agnostic replacements for the small set of DGL graph operations used
# by the SE(3)-Transformer.  These rely only on native PyTorch scatter/gather
# primitives (index_add_, index_select, scatter_reduce_) so the model runs on
# any PyTorch backend -- CUDA, ROCm/HIP, Intel XPU, or CPU -- without the DGL
# dependency, which ships prebuilt wheels only for specific CUDA versions and
# lacks support for newer architectures (e.g. NVIDIA Blackwell) and non-CUDA
# devices.
#
# Convention (matches DGL):
#   * A graph is a directed edge list; edge e runs from src[e] to dst[e].
#   * Messages flow src -> dst and are aggregated at the DESTINATION node.
#   * Edges (and their edata) are kept in the exact order supplied, which is
#     the edge-id order DGL guarantees for a graph built via
#     `dgl.graph((src, dst))` followed by `G.edata[...] = ...`.

from typing import Dict

import torch
from torch import Tensor


class Graph:
    """Minimal directed graph carrying an edge list and per-edge data.

    Provides only the slice of the ``DGLGraph`` API the SE(3)-Transformer
    actually touches: ``edges()``, ``num_nodes()``, the ``edata`` dict, and
    device movement via ``to()``.
    """

    def __init__(self, src: Tensor, dst: Tensor, num_nodes: int):
        assert src.shape == dst.shape, 'src and dst must have the same shape'
        self.src = src.long()
        self.dst = dst.long()
        self._num_nodes = int(num_nodes)
        self.edata: Dict[str, Tensor] = {}

    def edges(self):
        """Return (src, dst) endpoint tensors in edge-id order."""
        return self.src, self.dst

    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        return self.src.shape[0]

    @property
    def device(self):
        return self.src.device

    def to(self, device):
        self.src = self.src.to(device)
        self.dst = self.dst.to(device)
        self.edata = {k: v.to(device) for k, v in self.edata.items()}
        return self


def graph(src: Tensor, dst: Tensor, num_nodes: int) -> Graph:
    """Construct a :class:`Graph`. Mirrors ``dgl.graph((src, dst), num_nodes=N)``."""
    return Graph(torch.as_tensor(src), torch.as_tensor(dst), num_nodes)


def _scatter_add_nodes(num_nodes: int, index: Tensor, values: Tensor) -> Tensor:
    """Sum ``values`` (indexed per edge) into node buckets given by ``index``."""
    out = values.new_zeros((num_nodes,) + tuple(values.shape[1:]))
    out.index_add_(0, index, values)
    return out


def copy_e_sum(g: Graph, edge_feats: Tensor) -> Tensor:
    """Sum incoming edge features at each destination node.

    Equivalent to ``dgl.ops.copy_e_sum``. ``edge_feats`` has shape
    ``(num_edges, *feat)``; returns ``(num_nodes, *feat)``.
    """
    return _scatter_add_nodes(g.num_nodes(), g.dst, edge_feats)


def copy_e_mean(g: Graph, edge_feats: Tensor) -> Tensor:
    """Average incoming edge features at each destination node.

    Equivalent to ``dgl.ops.copy_e_mean``. Nodes with no incoming edges yield
    zeros (matching DGL), achieved by clamping the in-degree denominator to 1.
    """
    summed = _scatter_add_nodes(g.num_nodes(), g.dst, edge_feats)
    deg = edge_feats.new_zeros(g.num_nodes())
    deg.index_add_(0, g.dst, torch.ones_like(g.dst, dtype=edge_feats.dtype))
    deg = deg.clamp_(min=1.0).view((-1,) + (1,) * (edge_feats.dim() - 1))
    return summed / deg


def e_dot_v(g: Graph, edge_feats: Tensor, node_feats: Tensor) -> Tensor:
    """Per-edge dot product between edge features and the destination node.

    Equivalent to ``dgl.ops.e_dot_v``. The dot is taken over the last
    dimension, which is kept (size 1). ``edge_feats`` is ``(num_edges, *b, D)``
    and ``node_feats`` is ``(num_nodes, *b, D)``; returns ``(num_edges, *b, 1)``.
    """
    gathered = node_feats.index_select(0, g.dst)
    return (edge_feats * gathered).sum(dim=-1, keepdim=True)


def edge_softmax(g: Graph, scores: Tensor) -> Tensor:
    """Softmax of edge scores normalized over the incoming edges of each node.

    Equivalent to ``dgl.ops.edge_softmax`` / ``dgl.nn.functional.edge_softmax``.
    The per-node maximum is subtracted for numerical stability and detached, so
    gradients match a plain softmax (the constant shift cancels analytically).
    """
    num_nodes = g.num_nodes()
    dst = g.dst
    idx = dst.view((-1,) + (1,) * (scores.dim() - 1)).expand_as(scores)

    max_per_node = scores.new_full((num_nodes,) + tuple(scores.shape[1:]), float('-inf'))
    max_per_node.scatter_reduce_(0, idx, scores, reduce='amax', include_self=True)
    scores_max = max_per_node.index_select(0, dst).detach()

    exp_scores = torch.exp(scores - scores_max)
    denom = _scatter_add_nodes(num_nodes, dst, exp_scores).index_select(0, dst)
    return exp_scores / denom


def avg_pooling(g: Graph, node_feats: Tensor) -> Tensor:
    """Mean of node features over the whole (single) graph.

    Replacement for ``dgl.nn.pytorch.AvgPooling`` for the unbatched case that
    the SE(3)-Transformer uses here. Batched pooling would require per-graph
    node segments, which this lightweight Graph does not carry.
    """
    return node_feats.mean(dim=0, keepdim=True)


def max_pooling(g: Graph, node_feats: Tensor) -> Tensor:
    """Max of node features over the whole (single) graph.

    Replacement for ``dgl.nn.pytorch.MaxPooling`` for the unbatched case.
    """
    return node_feats.max(dim=0, keepdim=True).values
