#!/bin/bash

#SCRIPT="/home/src/rfantibody/scripts/rfdiffusion_inference.py"
SCRIPT="/home/src/rfantibody/rfdiffusion/rfdiffusion_inference.py"

poetry run python ${SCRIPT} \
    --config-name antibody \
    antibody.target_pdb=/home/demo/PDL1/7ounA_PDL1_T251.pdb \
    antibody.framework_pdb=/home/scripts/examples/example_inputs/hu-4D5-8_Fv.pdb \
    inference.ckpt_override_path=/home/weights/RFdiffusion_Ab.pt \
    'ppi.hotspot_res=[T289,T348,T356]' \
    'antibody.design_loops=[L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13]' \
    inference.num_designs=2 \
    inference.final_step=48 \
    diffuser.T=50 \
    inference.deterministic=False \
    inference.output_prefix=/home/demo/example_output/rfantibody/ab_des
