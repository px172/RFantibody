# XPU Port Notes

This document summarizes the current changes made to move RFantibody toward a vendor-neutral accelerator path, with emphasis on Intel XPU support and a pure PyTorch SE3 backend.

## What Changed

### 1. Pure PyTorch SE3 edge-list backend

I added a new graph abstraction and backend dispatch so SE3 can run without DGL as the execution backend:

- New edge-list graph wrapper: [include/SE3Transformer/se3_transformer/model/torch_graph.py](/home/px172/github/RFantibody-origin/include/SE3Transformer/se3_transformer/model/torch_graph.py)
- Backend-aware SE3 attention and aggregation:
  - [include/SE3Transformer/se3_transformer/model/layers/attention.py](/home/px172/github/RFantibody-origin/include/SE3Transformer/se3_transformer/model/layers/attention.py)
  - [include/SE3Transformer/se3_transformer/model/layers/convolution.py](/home/px172/github/RFantibody-origin/include/SE3Transformer/se3_transformer/model/layers/convolution.py)
- DGL graph creation in RFdiffusion and RF2 now routes through the shared graph helper:
  - [src/rfantibody/rfdiffusion/util_module.py](/home/px172/github/RFantibody-origin/src/rfantibody/rfdiffusion/util_module.py)
  - [src/rfantibody/rf2/network/util_module.py](/home/px172/github/RFantibody-origin/src/rfantibody/rf2/network/util_module.py)

The torch backend implements the subset of graph ops used by SE3:

- `edge_softmax`
- `e_dot_v`
- `copy_e_sum`
- `copy_e_mean`

The original DGL path is still preserved. The backend is selected by:

```bash
export RFANTIBODY_SE3_BACKEND=torch
```

If not set, the code prefers DGL when available.

### 2. Device abstraction for CUDA / XPU / CPU

I added a small device helper so the CLI and inference code can choose between CUDA, Intel XPU, and CPU without hard-coding NVIDIA assumptions:

- [src/rfantibody/util/device.py](/home/px172/github/RFantibody-origin/src/rfantibody/util/device.py)

Current behavior:

- If `RFANTIBODY_DEVICE` is set, it is used directly.
- Otherwise CUDA is preferred.
- If CUDA is unavailable and `torch.xpu` is available, XPU is selected.
- Otherwise CPU is used.

Examples:

```bash
export RFANTIBODY_DEVICE=xpu
export RFANTIBODY_SE3_BACKEND=torch
```

### 3. RFdiffusion runner updates

RFdiffusion now uses the shared device selector and prints a backend-appropriate device label in output metadata:

- [src/rfantibody/rfdiffusion/inference/model_runners.py](/home/px172/github/RFantibody-origin/src/rfantibody/rfdiffusion/inference/model_runners.py)
- [scripts/rfdiffusion_inference.py](/home/px172/github/RFantibody-origin/scripts/rfdiffusion_inference.py)

### 4. RF2 and ProteinMPNN device selection cleanup

These entry points were updated to use the same device helper:

- [scripts/rf2_predict.py](/home/px172/github/RFantibody-origin/scripts/rf2_predict.py)
- [src/rfantibody/rf2/modules/model_runner.py](/home/px172/github/RFantibody-origin/src/rfantibody/rf2/modules/model_runner.py)
- [scripts/proteinmpnn_interface_design.py](/home/px172/github/RFantibody-origin/scripts/proteinmpnn_interface_design.py)

This does not make them fully XPU-validated, but it removes the hard-coded `cuda:0` default path.

### 5. NVTX fallback for non-CUDA environments

I added a small profiling shim so SE3 modules do not fail on import when `torch.cuda.nvtx` is unavailable:

- [include/SE3Transformer/se3_transformer/runtime/profiling.py](/home/px172/github/RFantibody-origin/include/SE3Transformer/se3_transformer/runtime/profiling.py)

And I swapped the SE3 module imports to use it:

- [include/SE3Transformer/se3_transformer/model/basis.py](/home/px172/github/RFantibody-origin/include/SE3Transformer/se3_transformer/model/basis.py)
- [include/SE3Transformer/se3_transformer/model/layers/norm.py](/home/px172/github/RFantibody-origin/include/SE3Transformer/se3_transformer/model/layers/norm.py)

### 6. Hydra compatibility for Python 3.14 XPU venvs

The local XPU venv uses Python 3.14. Hydra 1.3.2 fails during argparse help validation before RFdiffusion starts because Hydra passes a lazy shell-completion help object instead of a plain string.

I added a small compatibility shim:

- [src/rfantibody/util/hydra_compat.py](/home/px172/github/RFantibody-origin/src/rfantibody/util/hydra_compat.py)

It is imported by:

- [scripts/rfdiffusion_inference.py](/home/px172/github/RFantibody-origin/scripts/rfdiffusion_inference.py)
- [scripts/rf2_predict.py](/home/px172/github/RFantibody-origin/scripts/rf2_predict.py)

The patch only activates on Python 3.14 or newer.

## XPU Venv Setup

The current working XPU environment is:

```bash
/home/px172/github/ProteinMPNN/.venv-xpu
```

Observed versions:

```text
Python 3.14.4
torch 2.12.0+xpu
torchvision 0.27.0+xpu
torchaudio 2.11.0+xpu
Intel(R) Arc(TM) Pro B70 Graphics
```

To create a similar environment from scratch:

```bash
python3.14 -m venv /home/px172/github/ProteinMPNN/.venv-xpu
source /home/px172/github/ProteinMPNN/.venv-xpu/bin/activate
python -m pip install --upgrade pip

# Install an Intel XPU-enabled PyTorch build that matches the local Intel GPU stack.
# The current tested environment has torch==2.12.0+xpu.
# Use the PyTorch/Intel XPU wheel source appropriate for this machine.
python -m pip install torch torchvision torchaudio

# RFantibody runtime dependencies needed by the CLI path.
# DGL is intentionally omitted for the XPU backend.
python -m pip install \
  e3nn \
  hydra-core \
  icecream \
  opt-einsum \
  biotite \
  pyrsistent \
  click \
  pandas
```

Do not use the project default dependency set for this XPU venv as-is. `pyproject.toml` is still pinned to Python 3.10, CUDA PyTorch, CUDA Python, and DGL. For XPU testing, use `PYTHONPATH` instead of installing the package:

```bash
source /home/px172/github/ProteinMPNN/.venv-xpu/bin/activate
export PYTHONPATH=/home/px172/github/RFantibody-origin/include/SE3Transformer:/home/px172/github/RFantibody-origin/src:$PYTHONPATH
export RFANTIBODY_ROOT=/home/px172/github/RFantibody-origin
export RFANTIBODY_WEIGHTS=/home/px172/github/RFantibody/weights
export RFANTIBODY_DEVICE=xpu
export RFANTIBODY_SE3_BACKEND=torch
```

Quick device check:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.xpu.is_available())
print(torch.xpu.get_device_name(0))
PY
```

## Verification So Far

I verified the new backend path with hand-run smoke comparisons in a working CUDA/DGL environment and with XPU smoke tests on the local Intel Arc Pro B70.

CPU / CUDA-stack environment checks:

- Small standalone `SE3Transformer` comparison between DGL and torch backend matched within floating point noise.
- RFdiffusion `Str2Str` comparison matched exactly on the test input used.
- RF2 `Str2Str` comparison matched exactly on the test input used.

XPU hardware checks:

- `xpu-smi discovery` sees `Intel(R) Arc(TM) Pro B70 Graphics`.
- `/home/px172/github/ProteinMPNN/.venv-xpu` has `torch 2.12.0+xpu`.
- `torch.xpu.is_available()` returns `True` outside the Codex sandbox.
- Basic XPU tensor matmul works.
- Pure PyTorch `TorchGraph` ops run on XPU.
- Small `SE3Transformer` forward runs on XPU with `RFANTIBODY_SE3_BACKEND=torch`.
- RFdiffusion `Str2Str` forward runs on XPU with `RFANTIBODY_SE3_BACKEND=torch`.
- RF2 `Str2Str` forward runs on XPU with `RFANTIBODY_SE3_BACKEND=torch`.
- Full RFdiffusion Click CLI runs end to end on XPU for a one-design smoke test.
- Full RF2 Click CLI runs end to end on XPU for a one-input smoke test.

The local `RFantibody-origin` `.venv` is currently not populated with `torch`, `dgl`, `e3nn`, or `pytest`, so I did not run pytest from that environment. The XPU smoke tests used `/home/px172/github/ProteinMPNN/.venv-xpu` with `PYTHONPATH` pointed at this repo.

Full RFdiffusion CLI smoke command:

```bash
source /home/px172/github/ProteinMPNN/.venv-xpu/bin/activate
export PYTHONPATH=/home/px172/github/RFantibody-origin/include/SE3Transformer:/home/px172/github/RFantibody-origin/src:$PYTHONPATH
export RFANTIBODY_ROOT=/home/px172/github/RFantibody-origin
export RFANTIBODY_WEIGHTS=/home/px172/github/RFantibody/weights
export RFANTIBODY_DEVICE=xpu
export RFANTIBODY_SE3_BACKEND=torch

python - <<'PY'
from rfantibody.cli.inference import rfdiffusion

rfdiffusion.main(args=[
    '--target', 'test/rfdiffusion/inputs_for_test/rsv_site3.pdb',
    '--framework', 'test/rfdiffusion/inputs_for_test/h-NbBCII10.pdb',
    '--output', '/tmp/rfantibody_xpu_cli/nb_des',
    '--num-designs', '1',
    '--design-loops', 'L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13',
    '--hotspots', 'T305,T456',
    '--diffuser-t', '50',
    '--final-step', '49',
    '--deterministic',
    '--no-trajectory',
], standalone_mode=True)
PY
```

Observed result:

```text
Output: /tmp/rfantibody_xpu_cli/nb_des_0.pdb
Device seen in runtime tensors: xpu:0
Finished design in 0.25 minutes
```

### RFdiffusion step count note

There are two different `diffuser.T` values that can appear in the logs:

- The RFdiffusion inference CLI/config default is `diffuser.T=50`.
- The checkpoint records `diffuser.T=200` from the model configuration used during training.

At runtime, the CLI passes `diffuser.T` as a Hydra override. This is why logs can show the checkpoint value first:

```text
USING MODEL CONFIG: self._conf[diffuser][T] = 200
WARNING: You are changing diffuser.T from the value this model was trained with.
```

But the actual inference loop uses the runtime value. With the normal CLI default:

```bash
DIFFUSER_T=50
FINAL_STEP=1
```

the loop runs:

```text
50, 49, 48, ..., 1
```

The XPU smoke command above intentionally used:

```bash
DIFFUSER_T=50
FINAL_STEP=49
```

so it only ran steps `50` and `49`. This was meant to quickly validate the XPU backend, checkpoint loading, SE3 torch backend, and PDB writing. For the usual RFdiffusion inference length on this repo, use:

```bash
DIFFUSER_T=50 FINAL_STEP=1 scripts/rfdiffusion_xpu_example.sh
```

### RF2 XPU smoke test

After RFdiffusion produced a valid PDB, I used that PDB as RF2 input and ran the full RF2 Click CLI on XPU:

```bash
source /home/px172/github/ProteinMPNN/.venv-xpu/bin/activate
export PYTHONPATH=/home/px172/github/RFantibody-origin/include/SE3Transformer:/home/px172/github/RFantibody-origin/src:$PYTHONPATH
export RFANTIBODY_ROOT=/home/px172/github/RFantibody-origin
export RFANTIBODY_WEIGHTS=/home/px172/github/RFantibody/weights
export RFANTIBODY_DEVICE=xpu
export RFANTIBODY_SE3_BACKEND=torch

python - <<'PY'
from rfantibody.cli.inference import rf2

rf2.main(args=[
    '--input-pdb', '/tmp/rfantibody_xpu_cli_newline/nb_des_0.pdb',
    '--output-dir', '/tmp/rfantibody_rf2_xpu_smoke',
    '--num-recycles', '1',
    '--seed', '0',
    '--no-cautious',
], standalone_mode=True)
PY
```

Observed result:

```text
[RF2] Processing: nb_des_0
[RF2]   Cycle 1/2 - pLDDT: 0.868
[RF2]   Cycle 2/2 - pLDDT: 0.871
[RF2] Completed: nb_des_0 - Best pLDDT: 0.871
Output: /tmp/rfantibody_rf2_xpu_smoke/nb_des_0_best.pdb
```

The output PDB does not contain doubled blank lines. Note that RF2 currently loops over `range(num_recycles + 1)`, so `--num-recycles 1` reports two cycles in the log.

## Current Limitations

### 1. XPU validation is still smoke-level

The full RFdiffusion and RF2 CLIs now run end to end on Intel Arc Pro B70 for smoke tests. The remaining validation is:

- Run larger `num_designs` batches.
- Run the default `final_step` settings instead of the shortened smoke test.
- Run RF2 with the default recycle count on a small batch.
- Compare deterministic output against CUDA for the same seed and backend assumptions.
- Measure memory use and throughput on Arc Pro B70.

### 2. Some code still assumes CUDA in non-SE3 areas

There are still CUDA-specific assumptions elsewhere in the repo, especially:

- some profiling and memory calls
- some CLI defaults in RF2 and ProteinMPNN
- the existing Docker / Apptainer images and dependency pins

These are now easier to isolate, but they are not all removed yet.

### 3. Environment setup still reflects the original NVIDIA stack

The project is still pinned around CUDA 11.8 wheels in the packaging files and NVIDIA base images. The code path is more portable now, but the packaging layer still needs a follow-up pass if the goal is a first-class XPU distribution.

The current XPU venv is intentionally minimal. It includes the RFdiffusion CLI dependencies needed for the smoke test, but it does not install RFantibody through `pyproject.toml` because the project metadata still targets Python 3.10 and CUDA.

## Suggested Next Steps

1. Run a longer RFdiffusion XPU test with default `final_step` and more designs.
2. Split packaging into backend-specific installs for CUDA and XPU.
3. Decide whether to keep the DGL backend as the default for CUDA machines or switch the default to `torch` once Intel validation is solid.
4. Add CI or documented manual tests for `RFANTIBODY_DEVICE=xpu` and `RFANTIBODY_SE3_BACKEND=torch`.
