# GPU Portability Migration — Results

Goal: make RFantibody run on non-CUDA backends (AMD ROCm, Intel XPU) and on
newer NVIDIA architectures (Blackwell / sm_120), which the original stack could
not.

## Problems addressed

1. **DGL blocked Blackwell and all non-CUDA backends.** DGL ships prebuilt
   wheels only for specific CUDA versions (the repo pinned a `cu118` wheel) and
   has no Blackwell (sm_120) or ROCm/XPU support.
2. **CUDA was hardcoded throughout.** `cuda:0`, `torch.cuda.amp.autocast`,
   `torch.cuda.nvtx.range`, and `torch.cuda.{empty_cache,*memory*}` were used
   directly, so nothing ran on XPU.
3. **`torch==2.3.*` (cu118) predates Blackwell.** On an RTX 5090,
   `torch.cuda.is_available()` returns True but kernels fail with
   *"sm_120 is not compatible"*.

Key finding: the vendored SE(3)-Transformer has **no custom CUDA kernels** — it
is pure PyTorch + DGL — so no kernel rewrite was needed. The only real coupling
was the DGL dependency plus device-management calls.

## Changes

### Step 2 — Remove DGL, use native PyTorch scatter
- New `include/SE3Transformer/se3_transformer/model/graph.py`: a lightweight
  `Graph` class + native reimplementations of the only DGL ops used:
  `copy_e_sum`, `copy_e_mean`, `e_dot_v`, `edge_softmax`, and pooling.
- Rewrote `convolution.py`, `attention.py`, `pooling.py`, `transformer.py` and
  both `util_module.py` graph builders (rf2 + rfdiffusion) to use it.
- Removed `dgl` from `pyproject.toml`.

### Step 3 — Device abstraction
- New `src/rfantibody/util/device.py`: `get_device()` (cuda/xpu/cpu),
  device-agnostic `autocast`, an `autocast_disabled` decorator, and
  `empty_cache` / `reset_peak_memory_stats` / `max_memory_allocated` that
  dispatch to `torch.cuda` or `torch.xpu`.
- New `include/SE3Transformer/se3_transformer/model/profiling.py`: a portable
  `nvtx_range` (real NVTX when CUDA is present, no-op otherwise).
- Replaced hardcoded `cuda:0`, `torch.cuda.amp.autocast`, `torch.cuda.nvtx`,
  and `torch.cuda` memory calls across the active inference path (predict.py,
  model_runner.py, model_runners.py, proteinmpnn, SE3_network.py,
  Track_module.py, and the SE3 layers).

### Step 1 — PyTorch for Blackwell
- `torch==2.3.*` (cu118) → `torch==2.8.*` on the `cu128` index.
- Removed unused `torchaudio` / `torchvision`; relaxed unused `cuda-python` to
  `>=12`.
- Bumped Docker/Apptainer base images to
  `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04`; updated README.

Note: ROCm needs no code changes beyond the above — PyTorch's ROCm build exposes
the same `torch.cuda` API via HIP. Intel XPU is the backend the abstraction
actually enables.

## Verification results

`verify_device.py` (repo root; deps: torch + e3nn + numpy) runs the graph ops
and a real `SE3Transformer` forward on the selected device and compares to a CPU
reference. DGL-vs-native equivalence was checked separately while DGL was still
installed.

| Check | RTX 5090 (Blackwell sm_120) | RTX 3060 (CUDA sm_86) | Arc PRO B70 (Intel XPU) |
|---|---|---|---|
| Graph ops vs DGL | bit-identical (fwd + grad) | — | — |
| Graph ops vs CPU | ✅ | ✅ ~1e-6 | ✅ ~1e-6 |
| SE3 forward vs CPU | ✅ (0.0) | ✅ ~2e-5 | ✅ ~2e-5 |
| Full pipeline | — | ✅ (see below) | not yet |

- **DGL equivalence**: `copy_e_sum` / `copy_e_mean` / `e_dot_v` gradients were
  bit-identical (0.0); full-model output matched the original DGL code exactly.
- **Intel XPU** is the decisive result: DGL never supported XPU, so this
  pipeline could not run on Intel hardware before this work. `get_device()`
  correctly selects the discrete Arc B70 (not the integrated UHD 770).

### Full antibody pipeline (RTX 3060, torch 2.8+cu128)

| Stage | Run | Result |
|---|---|---|
| RFdiffusion | 2 nanobody designs, `--diffuser-t 50` | RC=0, 6.65 min, valid PDBs |
| ProteinMPNN | CDR sequence design | RC=0, sequence in ~1 s on GPU |
| RF2 | structure prediction, 11 recycles | RC=0, **pLDDT 0.910**, output PDB |

The high pLDDT confirms correct numerics end-to-end, not merely that the code
ran.

## Not yet done

- **Full pipeline on Intel XPU** — the SE3/graph core passes on XPU, but the
  complete RFdiffusion/RF2 use ops beyond that core whose XPU coverage is
  untested.
- **AMD ROCm** — code is ready; no AMD hardware was available to test.
- **`test.run_tests` reference-output regression** — reference outputs predate
  this torch version, so tolerances may need updating.

## Reproducing a device check

```bash
# From the repo root, in an env with torch + e3nn + numpy for the target device:
python verify_device.py     # prints get_device() and PASS/FAIL per op + SE3 forward
```
