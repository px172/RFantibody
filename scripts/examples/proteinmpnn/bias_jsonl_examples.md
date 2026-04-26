# ProteinMPNN Bias JSONL Examples

This note documents the two bias inputs exposed by `uv run proteinmpnn` in
RFantibody:

- `--bias_AA_jsonl`
- `--bias_by_res_jsonl`

`bias_AA_jsonl` is a global amino-acid bias.
`bias_by_res_jsonl` is an RFantibody-specific shared per-position bias format
that is applied to every structure in the batch.

## Helper Scripts

This repo also includes two helper scripts for generating these files:

- [make_bias_AA.py](/work/px172/github/RFantibody_2026v1/helper_scripts/make_bias_AA.py:1)
- [make_bias_by_res_jsonl.py](/work/px172/github/RFantibody_2026v1/helper_scripts/make_bias_by_res_jsonl.py:1)

## Shared Rules

- Amino-acid column order is:

```text
ACDEFGHIKLMNPQRSTVWYX
```

- Positive values make an amino acid more likely
- Negative values make an amino acid less likely
- `0.0` means no bias
- Bias is added before softmax and is divided by temperature

The sampling code applies:

```text
softmax(logits + global_bias / temperature + per_res_bias / temperature)
```

With RFantibody's default `--temperature 0.1`, even moderate bias values are
quite strong.

## `bias_AA_jsonl`

### Format

Top-level dictionary:

```json
{
  "A": -1.1,
  "F": 0.7,
  "W": 0.4,
  "C": -2.0
}
```

Meaning:

- `A: -1.1`: globally discourage alanine
- `F: 0.7`: globally encourage phenylalanine
- `W: 0.4`: globally encourage tryptophan
- `C: -2.0`: strongly discourage cysteine

Any amino acid not listed gets `0.0`.

### Example File

See [bias_AA_example.jsonl](</work/px172/github/RFantibody_2026v1/scripts/examples/proteinmpnn/bias_AA_example.jsonl:1>).

### Helper Script

```bash
uv run python helper_scripts/make_bias_AA.py \
  --AA_list "A F W C" \
  --bias_list "-1.1 0.7 0.4 -2.0" \
  --output_path bias_AA.jsonl
```

### Example Command

```bash
uv run proteinmpnn \
  --input-dir scripts/examples/proteinmpnn/example_inputs \
  --output-dir /tmp/proteinmpnn_bias_demo \
  --bias_AA_jsonl bias_AA.jsonl
```

## `bias_by_res_jsonl`

### RFantibody Format

RFantibody uses a shared chain-level format:

```json
{
  "H": [[21 numbers], [21 numbers], ...],
  "L": [[21 numbers], [21 numbers], ...]
}
```

Meaning:

- top-level keys are chain IDs
- each value is a matrix with shape `[chain_length, 21]`
- the same matrix is applied to every structure processed in that ProteinMPNN run

Each row corresponds to one residue in the chain, in chain order:

- row `0` = residue 1 in that chain
- row `1` = residue 2 in that chain
- row `n-1` = residue `n`

Each row has 21 columns in this order:

```text
A C D E F G H I K L M N P Q R S T V W Y X
```

### What Chains Need Entries

In RFantibody's antibody interface design flow:

- designed chains are usually `H` and `L`
- target chain `T` is visible/fixed, not sampled

So in practice:

- you usually provide entries for `H` and/or `L`
- you usually do not need to provide `T`

The matrix should at least cover all masked/design chains.

### Toy Example

See [bias_by_res_toy_example.jsonl](</work/px172/github/RFantibody_2026v1/scripts/examples/proteinmpnn/bias_by_res_toy_example.jsonl:1>).

That toy file encodes:

- chain `H`, residue 2: favor `Y` and slightly favor `F`
- chain `H`, residue 3: discourage `G`, favor `W`
- chain `L`, residue 1: favor acidic residues `D` and `E`
- chain `L`, residue 3: strongly discourage `C`

## Making a Real `bias_by_res_jsonl`

The main requirement is that each chain matrix must match the full chain length.

For the bundled ProteinMPNN example input
[ab_rfdiffusion_output.pdb](</work/px172/github/RFantibody_2026v1/scripts/examples/proteinmpnn/example_inputs/ab_rfdiffusion_output.pdb:1>),
the chain lengths are:

- `H`: 121
- `L`: 106
- `T`: 251

Since `T` is not designed, you typically only need matrices for `H` and `L`.

### Recommended Workflow

1. Start from all zeros for each designed chain.
2. Set only the positions and amino acids you want to bias.
3. Save the result as JSON.

### Sparse-to-Full Generator Example

```python
import json

alphabet = "ACDEFGHIKLMNPQRSTVWYX"

chain_lengths = {"H": 121, "L": 106}
sparse = {
    "H": {
        31: {"Y": 2.0, "F": 0.8},
        52: {"G": -1.5, "W": 1.2},
    },
    "L": {
        30: {"D": 1.5, "E": 0.7},
        90: {"C": -2.0},
    },
}

result = {}

for chain, chain_len in chain_lengths.items():
    matrix = [[0.0] * len(alphabet) for _ in range(chain_len)]
    for pos_1idx, aa_biases in sparse.get(chain, {}).items():
        row = matrix[pos_1idx - 1]
        for aa, value in aa_biases.items():
            row[alphabet.index(aa)] = value
    result[chain] = matrix

with open("bias_by_res.json", "w") as f:
    json.dump(result, f)
```

### Helper Script

The easier way to build a `bias_by_res_jsonl` file is to use
[make_bias_by_res_jsonl.py](/work/px172/github/RFantibody_2026v1/helper_scripts/make_bias_by_res_jsonl.py:1).

Example using a PDB to infer chain lengths:

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

Example using explicit chain lengths:

```bash
uv run python helper_scripts/make_bias_by_res_jsonl.py \
  --chain_lengths H:121 L:106 \
  --set H:31:Y:2.0 \
  --set H:31:F:0.8 \
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

### What Happens at Unspecified Positions

An unspecified position does not mean "keep the original amino acid".
It only means "no extra bias is applied at this position".

Behavior depends on whether the position is designable:

- position is fixed:
  - it stays as the original residue
- position is designable, but bias row is all zeros:
  - ProteinMPNN still redesigns it
  - it is just sampled without extra bias
- position is designable, and bias row has nonzero values:
  - ProteinMPNN redesigns it with those local preferences

In RFantibody's `uv run proteinmpnn` flow, whether a position is designable is
mainly controlled by `--loops`.

### Value Range

There is no hard-coded numeric range check in the current implementation.
Each bias value is added directly to the amino-acid logits before softmax.

Practical interpretation:

- `0.0`: no effect
- positive value: favor that amino acid
- negative value: discourage that amino acid

The effective strength is scaled by temperature:

```text
effective_bias = bias / temperature
```

Recommended starting ranges:

- `±0.2` to `±0.5`: mild preference
- `±0.5` to `±1.0`: moderate to strong preference
- `±1.0` to `±2.0`: aggressive preference
- above `±3.0`: usually too strong unless you intentionally want near-forced behavior

## Combining Both Bias Types

You can use both files at once:

```bash
uv run proteinmpnn \
  --input-dir scripts/examples/proteinmpnn/example_inputs \
  --output-dir /tmp/proteinmpnn_bias_demo \
  --bias_AA_jsonl scripts/examples/proteinmpnn/bias_AA_example.jsonl \
  --bias_by_res_jsonl bias_by_res.jsonl
```

Interpretation:

- `bias_AA_jsonl` changes the global composition tendency
- `bias_by_res_jsonl` adds local overrides at specific positions for every structure in the batch
