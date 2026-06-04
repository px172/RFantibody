#!/usr/bin/env bash
set -euo pipefail

RFANTIBODY_ROOT="${RFANTIBODY_ROOT:-/home/px172/github/RFantibody-origin}"
RFANTIBODY_WEIGHTS="${RFANTIBODY_WEIGHTS:-/home/px172/github/RFantibody/weights}"
XPU_VENV="${XPU_VENV:-/home/px172/github/ProteinMPNN/.venv-xpu}"

TARGET_PDB="${TARGET_PDB:-${RFANTIBODY_ROOT}/test/rfdiffusion/inputs_for_test/rsv_site3.pdb}"
FRAMEWORK_PDB="${FRAMEWORK_PDB:-${RFANTIBODY_ROOT}/test/rfdiffusion/inputs_for_test/h-NbBCII10.pdb}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/tmp/rfantibody_xpu_cli/nb_des}"

NUM_DESIGNS="${NUM_DESIGNS:-1}"
DIFFUSER_T="${DIFFUSER_T:-50}"
FINAL_STEP="${FINAL_STEP:-49}"
DESIGN_LOOPS="${DESIGN_LOOPS:-L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13}"
HOTSPOTS="${HOTSPOTS:-T305,T456}"

source "${XPU_VENV}/bin/activate"

export RFANTIBODY_ROOT
export RFANTIBODY_WEIGHTS
export RFANTIBODY_DEVICE="${RFANTIBODY_DEVICE:-xpu}"
export RFANTIBODY_SE3_BACKEND="${RFANTIBODY_SE3_BACKEND:-torch}"
export PYTHONPATH="${RFANTIBODY_ROOT}/include/SE3Transformer:${RFANTIBODY_ROOT}/src:${PYTHONPATH:-}"

python - <<PY
import torch

print(f"torch: {torch.__version__}")
print(f"xpu available: {torch.xpu.is_available()}")
if torch.xpu.is_available():
    print(f"xpu device: {torch.xpu.get_device_name(0)}")
PY

python - <<PY
from rfantibody.cli.inference import rfdiffusion

rfdiffusion.main(args=[
    "--target", "${TARGET_PDB}",
    "--framework", "${FRAMEWORK_PDB}",
    "--output", "${OUTPUT_PREFIX}",
    "--num-designs", "${NUM_DESIGNS}",
    "--design-loops", "${DESIGN_LOOPS}",
    "--hotspots", "${HOTSPOTS}",
    "--diffuser-t", "${DIFFUSER_T}",
    "--final-step", "${FINAL_STEP}",
    "--deterministic",
    "--no-trajectory",
], standalone_mode=True)
PY
