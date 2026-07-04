"""Self-contained portability check: run the DGL-free graph ops and a real
SE3Transformer on whatever accelerator get_device() selects (CUDA / Intel XPU /
CPU) and compare against a CPU reference within fp tolerance.

Deps: torch + e3nn + numpy only. Run from the repo root:
    python verify_device.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "include", "SE3Transformer"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch

from rfantibody.util.device import get_device
from se3_transformer.model import SE3Transformer
from se3_transformer.model.fiber import Fiber
from se3_transformer.model.graph import (
    graph, copy_e_sum, copy_e_mean, e_dot_v, edge_softmax,
)

DEV = get_device()
print(f"torch {torch.__version__}")
print(f"get_device() -> {DEV}")
if DEV.type == "cuda":
    print("  cuda:", torch.cuda.get_device_name(0),
          "sm_%d%d" % torch.cuda.get_device_capability(0))
elif DEV.type == "xpu":
    print("  xpu:", torch.xpu.get_device_name(0))


def cmp(name, a, b, atol=1e-3, rtol=1e-3):
    a, b = a.float().cpu(), b.float().cpu()
    d = (a - b).abs().max().item()
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:20s} max_abs_diff={d:.3e}")
    return ok


torch.manual_seed(0)
N, E = 40, 300
src = torch.randint(0, N, (E,))
dst = torch.randint(0, N, (E,))

all_ok = True

# ---- graph ops: device vs CPU ----
print("\n[graph ops: device vs cpu]")
g_cpu = graph(src, dst, N)
g_dev = graph(src.clone(), dst.clone(), N).to(DEV)

ef3 = torch.randn(E, 5, 3)
all_ok &= cmp("copy_e_sum", copy_e_sum(g_cpu, ef3), copy_e_sum(g_dev, ef3.to(DEV)))
all_ok &= cmp("copy_e_mean", copy_e_mean(g_cpu, ef3), copy_e_mean(g_dev, ef3.to(DEV)))
ev = torch.randn(E, 4, 7); nv = torch.randn(N, 4, 7)
all_ok &= cmp("e_dot_v", e_dot_v(g_cpu, ev, nv), e_dot_v(g_dev, ev.to(DEV), nv.to(DEV)))
sc = torch.randn(E, 4)
all_ok &= cmp("edge_softmax", edge_softmax(g_cpu, sc), edge_softmax(g_dev, sc.to(DEV)))

# ---- full SE3Transformer: device vs CPU (same weights/inputs/graph) ----
print("\n[SE3Transformer forward: device vs cpu]")
torch.manual_seed(1234)
se3 = SE3Transformer(
    num_layers=2,
    fiber_in=Fiber({0: 32, 1: 3}),
    fiber_hidden=Fiber.create(2, 32),
    fiber_out=Fiber({0: 16, 1: 2}),
    num_heads=4, channels_div=4,
    fiber_edge=Fiber({0: 32}),
    use_layer_norm=True,
).eval()

L = 24
G = graph(torch.repeat_interleave(torch.arange(L), L),
          torch.arange(L).repeat(L), L)          # fully connected LxL
G.edata["rel_pos"] = torch.randn(G.num_edges, 3)
node = {"0": torch.randn(L, 32, 1), "1": torch.randn(L, 3, 3)}
edge = {"0": torch.randn(G.num_edges, 32, 1)}


def run(dev):
    se3.to(dev)
    G.to(dev)
    nd = {k: v.to(dev) for k, v in node.items()}
    ed = {k: v.to(dev) for k, v in edge.items()}
    with torch.no_grad():
        out = se3(G, nd, ed)
    return {k: v.float().cpu() for k, v in out.items()}


cpu_out = run(torch.device("cpu"))
dev_out = run(DEV)
for k in cpu_out:
    all_ok &= cmp(f"type{k}", cpu_out[k], dev_out[k])

print("\nRESULT:", f"ALL PASS on {DEV.type}" if all_ok else "SOME FAILED")
sys.exit(0 if all_ok else 1)
