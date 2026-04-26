#!/usr/bin/env python3

import argparse
import json


ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Generate an RFantibody-compatible shared bias_by_res_jsonl file from "
            "simple CHAIN:POSITION:AA:VALUE assignments."
        ),
    )
    parser.add_argument("--output_path", required=True, help="Output JSONL path")
    parser.add_argument(
        "--pdb_path",
        default="",
        help="Optional PDB used to infer chain lengths from CA atoms",
    )
    parser.add_argument(
        "--chain_lengths",
        nargs="*",
        default=[],
        help="Optional explicit chain lengths like H:121 L:106; used if --pdb_path is not provided",
    )
    parser.add_argument(
        "--set",
        dest="bias_entries",
        action="append",
        default=[],
        help=(
            "Bias assignment in CHAIN:POSITION:AA:VALUE form. "
            "Repeat this option to add multiple assignments."
        ),
    )
    return parser.parse_args()


def infer_chain_lengths_from_pdb(pdb_path):
    chain_residues = {}
    with open(pdb_path) as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue

            chain = line[21:22]
            residue_id = line[22:27].strip()
            chain_residues.setdefault(chain, [])
            if not chain_residues[chain] or chain_residues[chain][-1] != residue_id:
                chain_residues[chain].append(residue_id)

    if not chain_residues:
        raise ValueError(f"No CA atoms found in PDB: {pdb_path}")

    return {chain: len(residues) for chain, residues in chain_residues.items()}


def parse_chain_lengths(raw_chain_lengths):
    chain_lengths = {}
    for item in raw_chain_lengths:
        try:
            chain, length = item.split(":")
        except ValueError as exc:
            raise ValueError(
                f"Invalid --chain_lengths entry '{item}'. Expected CHAIN:LENGTH"
            ) from exc
        chain = chain.strip()
        if len(chain) != 1:
            raise ValueError(f"Chain ID must be a single character: '{chain}'")
        chain_lengths[chain] = int(length)
    return chain_lengths


def parse_bias_entries(entries):
    parsed = []
    for item in entries:
        try:
            chain, pos, aa, value = item.split(":")
        except ValueError as exc:
            raise ValueError(
                f"Invalid --set entry '{item}'. Expected CHAIN:POSITION:AA:VALUE"
            ) from exc

        chain = chain.strip()
        aa = aa.strip().upper()
        position = int(pos)
        bias = float(value)

        if len(chain) != 1:
            raise ValueError(f"Chain ID must be a single character: '{chain}'")
        if aa not in ALPHABET:
            raise ValueError(
                f"Invalid amino acid '{aa}'. Allowed values are characters from {ALPHABET}"
            )
        if position < 1:
            raise ValueError(f"Position must be 1-indexed and positive: '{position}'")

        parsed.append((chain, position, aa, bias))
    return parsed


def main():
    args = parse_args()

    if args.pdb_path:
        chain_lengths = infer_chain_lengths_from_pdb(args.pdb_path)
    else:
        chain_lengths = parse_chain_lengths(args.chain_lengths)

    if not chain_lengths:
        raise ValueError("Provide either --pdb_path or at least one --chain_lengths entry")

    bias_entries = parse_bias_entries(args.bias_entries)

    bias_dict = {
        chain: [[0.0] * len(ALPHABET) for _ in range(chain_length)]
        for chain, chain_length in chain_lengths.items()
    }

    for chain, position, aa, bias in bias_entries:
        if chain not in bias_dict:
            raise ValueError(
                f"Chain '{chain}' was referenced in --set but is not present in the inferred/provided chain lengths"
            )
        chain_length = len(bias_dict[chain])
        if position > chain_length:
            raise ValueError(
                f"Position {position} is out of range for chain '{chain}' of length {chain_length}"
            )
        bias_dict[chain][position - 1][ALPHABET.index(aa)] = bias

    with open(args.output_path, "w") as handle:
        handle.write(json.dumps(bias_dict) + "\n")


if __name__ == "__main__":
    main()
