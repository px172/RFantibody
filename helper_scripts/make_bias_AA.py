#!/usr/bin/env python3

import argparse
import json


ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate a ProteinMPNN-compatible bias_AA_jsonl file.",
    )
    parser.add_argument("--output_path", required=True, help="Output JSONL path")
    parser.add_argument(
        "--AA_list",
        default="",
        help="Whitespace-separated list of amino acids to bias, e.g. 'A F W C'",
    )
    parser.add_argument(
        "--bias_list",
        default="",
        help="Whitespace-separated list of bias values, e.g. '-1.1 0.7 0.4 -2.0'",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    aa_list = [item.strip().upper() for item in args.AA_list.split() if item.strip()]
    bias_list = [float(item) for item in args.bias_list.split() if item.strip()]

    if len(aa_list) != len(bias_list):
        raise ValueError(
            f"AA_list length ({len(aa_list)}) does not match bias_list length ({len(bias_list)})"
        )

    invalid_aas = [aa for aa in aa_list if aa not in ALPHABET]
    if invalid_aas:
        raise ValueError(
            f"Invalid amino acids {invalid_aas}; allowed values are characters from {ALPHABET}"
        )

    bias_dict = {}
    for aa, bias in zip(aa_list, bias_list):
        bias_dict[aa] = bias

    with open(args.output_path, "w") as handle:
        handle.write(json.dumps(bias_dict) + "\n")


if __name__ == "__main__":
    main()
