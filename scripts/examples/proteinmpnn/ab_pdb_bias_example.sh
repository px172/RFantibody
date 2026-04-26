#!/bin/bash

# Example: Design antibody sequences using ProteinMPNN with bias inputs
#
# This example generates both a global bias_AA file and a per-position
# bias_by_res file, then runs the proteinmpnn CLI with those inputs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

OUTPUT_DIR="$SCRIPT_DIR/example_outputs_bias"
BIAS_AA_JSONL="$SCRIPT_DIR/example_outputs_bias_AA.jsonl"
BIAS_BY_RES_JSONL="$SCRIPT_DIR/example_outputs_bias_by_res.jsonl"
INPUT_PDB="$SCRIPT_DIR/example_inputs/ab_rfdiffusion_output.pdb"

uv run python "$ROOT_DIR/helper_scripts/make_bias_AA.py" \
    --AA_list "A F W C" \
    --bias_list "-1.1 0.7 0.4 -2.0" \
    --output_path "$BIAS_AA_JSONL"

uv run python "$ROOT_DIR/helper_scripts/make_bias_by_res_jsonl.py" \
    --pdb_path "$INPUT_PDB" \
    --set H:31:Y:2.0 \
    --set H:31:F:0.8 \
    --set H:52:W:1.2 \
    --set H:52:G:-1.5 \
    --set L:30:D:1.5 \
    --set L:30:E:0.7 \
    --set L:90:C:-2.0 \
    --output_path "$BIAS_BY_RES_JSONL"

uv run proteinmpnn \
    --input-dir "$SCRIPT_DIR/example_inputs" \
    --output-dir "$OUTPUT_DIR" \
    --bias_AA_jsonl "$BIAS_AA_JSONL" \
    --bias_by_res_jsonl "$BIAS_BY_RES_JSONL"
