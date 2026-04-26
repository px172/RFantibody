# Helper Scripts

This directory contains small utility scripts for generating ProteinMPNN bias
input files.

Currently included:

- `make_bias_AA.py`
- `make_bias_by_res_jsonl.py`

## `make_bias_AA.py`

Generates a ProteinMPNN-compatible `bias_AA_jsonl` file for global amino-acid
bias.

Output format:

```json
{"A": -1.1, "F": 0.7, "W": 0.4, "C": -2.0}
```

Example:

```bash
uv run python helper_scripts/make_bias_AA.py \
  --AA_list "A F W C" \
  --bias_list "-1.1 0.7 0.4 -2.0" \
  --output_path bias_AA.jsonl
```

Then use it with:

```bash
uv run proteinmpnn \
  --input-dir scripts/examples/proteinmpnn/example_inputs \
  --output-dir /tmp/proteinmpnn_bias_demo \
  --bias_AA_jsonl bias_AA.jsonl
```

Notes:

- `AA_list` and `bias_list` must have the same length
- allowed amino acids follow `ACDEFGHIKLMNPQRSTVWYX`
- positive values favor an amino acid
- negative values discourage an amino acid

## `make_bias_by_res_jsonl.py`

Generates an RFantibody-compatible shared `bias_by_res_jsonl` file from simple
`CHAIN:POSITION:AA:VALUE` assignments.

This script expands sparse per-position rules into the full matrix format
required by RFantibody:

```json
{
  "<chain>": [[21 values], [21 values], ...]
}
```

The amino-acid column order is:

```text
ACDEFGHIKLMNPQRSTVWYX
```

### Example Using a PDB

This is the easiest mode. The script infers chain lengths from CA atoms in the
PDB.

```bash
uv run python helper_scripts/make_bias_by_res_jsonl.py \
  --pdb_path scripts/examples/proteinmpnn/example_inputs/ab_rfdiffusion_output.pdb \
  --set H:31:Y:2.0 \
  --set H:31:F:0.8 \
  --set H:52:W:1.2 \
  --set H:52:G:-1.5 \
  --set L:30:D:1.5 \
  --set L:30:E:0.7 \
  --set L:90:C:-2.0 \
  --output_path bias_by_res.jsonl
```

Then use it with:

```bash
uv run proteinmpnn \
  --input-dir scripts/examples/proteinmpnn/example_inputs \
  --output-dir /tmp/proteinmpnn_bias_demo \
  --bias_by_res_jsonl bias_by_res.jsonl
```

### Example Using Explicit Chain Lengths

Use this mode when you do not want to provide a PDB.

```bash
uv run python helper_scripts/make_bias_by_res_jsonl.py \
  --chain_lengths H:121 L:106 \
  --set H:31:Y:2.0 \
  --set H:31:F:0.8 \
  --set L:90:C:-2.0 \
  --output_path bias_by_res.jsonl
```

### `--set` Format

Each `--set` entry must be:

```text
CHAIN:POSITION:AA:VALUE
```

Example:

- `H:31:Y:2.0` means chain `H`, residue `31`, bias amino acid `Y` by `+2.0`
- `L:90:C:-2.0` means chain `L`, residue `90`, discourage `C` with `-2.0`

Rules:

- `POSITION` is 1-indexed
- `AA` must be one character from `ACDEFGHIKLMNPQRSTVWYX`
- the same position can be specified multiple times for different amino acids
- chain length must be known either from `--pdb_path` or `--chain_lengths`

## Practical Advice

- start with small magnitudes such as `0.3` to `1.0`
- RFantibody's default ProteinMPNN temperature is `0.1`, so bias values are
  quite strong
- use `bias_AA_jsonl` for global composition trends
- use shared `bias_by_res_jsonl` for local, position-specific preferences across all structures in the batch

## Related Docs

For a fuller explanation of formats and behavior, see:

- [ProteinMPNN bias examples](</work/px172/github/RFantibody_2026v1/scripts/examples/proteinmpnn/bias_jsonl_examples.md:1>)
