#!/bin/bash

RF2_OUTPUT="/home/demo/example_output/rf2"

mkdir -p ${RF2_OUTPUT}

poetry run python /home/scripts/rf2_predict.py \
    input.pdb_dir=/home/demo/example_output/mpnn/ \
    output.pdb_dir=${RF2_OUTPUT} \
    model.model_weights=/home/weights/RF2_ab.pt \
