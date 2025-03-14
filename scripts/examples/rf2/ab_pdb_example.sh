#!/bin/bash

mkdir -p /home/scripts/examples/rf2/example_outputs

poetry run python /home/scripts/rf2_predict.py \
    input.pdb_dir=/home/scripts/examples/rf2/example_inputs \
    output.pdb_dir=/home/scripts/examples/rf2/example_outputs \
    model.model_weights=/home/weights/RF2_ab.pt \
