import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

import rfantibody.proteinmpnn.util_protein_mpnn as mpnn_util
from rfantibody.proteinmpnn.sample_features import SampleFeatures
from rfantibody.proteinmpnn.struct_manager import StructManager

#################################
# Parse Arguments
#################################

parser = argparse.ArgumentParser()

# I/O Arguments
parser.add_argument("-pdbdir", type=str, default="", help='The name of a directory of pdbs to run through the model')
parser.add_argument("-quiver", type=str, default="", help='The name of a quiver file to run this metric on.')

parser.add_argument("-outquiver", type=str, default="",
                    help="The name of the quiver file to which output structs will be written")
parser.add_argument("-outpdbdir", type=str, default="outputs",
                    help='The directory to which the output PDB files will be written')
parser.add_argument("-runlist", type=str, default='',
                    help="The path of a list of pdb tags to run (default: ''; Run all PDBs")
parser.add_argument("-checkpoint_name", type=str, default='check.point',
                    help="The name of a file where tags which have finished will be written (default: check.point)")
parser.add_argument("-debug", action="store_true", default=False,
                    help='When active, errors will cause the script to crash and the error message ' + \
                         'to be printed out (default: False)')
parser.add_argument("-deterministic", action="store_true", default=False,
                    help='Enable deterministic mode by setting fixed seeds and deterministic algorithms (default: False)')

# Design Arguments
parser.add_argument("-loop_string", type=str, default='H1,H2,H3,L1,L2,L3',
                    help='The list of loops which you wish to design')
parser.add_argument("-seqs_per_struct", type=int, default="1",
                    help="The number of sequences to generate for each structure (default: 1)")

# ProteinMPNN Specific Arguments
default_ckpt = os.path.join( os.path.dirname(__file__), '/home/weights/ProteinMPNN_v48_noise_0.2.pt')
parser.add_argument("-checkpoint_path", type=str, default=default_ckpt)
parser.add_argument("-temperature", type=float, default=0.000001, help='An a3m file containing the MSA of your target')
parser.add_argument("-augment_eps", type=float, default=0,
                    help='The variance of random noise to add to the atomic coordinates (default 0)')
parser.add_argument("-protein_features", type=str, default='full',
                    help='What type of protein features to input to ProteinMPNN (default: full)')
parser.add_argument("-omit_AAs", type=str, default='CX',
                    help='A string of all residue types (one letter case-insensitive) that you would not like to ' + \
                         'use for design. Letters not corresponding to residue types will be ignored')
parser.add_argument("-bias_AA_jsonl", "--bias_AA_jsonl", type=str, default='',
                    help="Path to a dictionary which specifies AA composition bias, e.g. {\"A\": -1.1, \"F\": 0.7}")
parser.add_argument("-bias_by_res_jsonl", "--bias_by_res_jsonl", type=str, default='',
                    help="Path to dictionary with shared per-position amino-acid bias in RFantibody format, e.g. {\"H\": [[...]], \"L\": [[...]]}")
parser.add_argument("-num_connections", type=int, default=48,
                    help='Number of neighbors each residue is connected to, default 48, higher number leads to ' + \
                         'better interface design but will cost more to run the model.')
parser.add_argument("-allow_x", action="store_true", default=False,
                    help='Allow X (unknown) residues in output. Useful for debugging to see exactly which positions were designed.')

args = parser.parse_args(sys.argv[1:])

# Set up deterministic mode if requested
if args.deterministic:
    print("Setting up deterministic mode with fixed seeds")
    # Set fixed seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Enable deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_jsonl_dict(jsonl_path: str):
    if not jsonl_path:
        return None

    with open(jsonl_path, 'r') as json_file:
        raw_text = json_file.read().strip()

    if not raw_text:
        return None

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        parsed_dict = None
        for json_str in raw_text.splitlines():
            json_str = json_str.strip()
            if not json_str:
                continue
            parsed_dict = json.loads(json_str)
        return parsed_dict


def _build_bias_AAs_np(bias_AA_dict):
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    return np.array([bias_AA_dict.get(aa, 0.0) for aa in alphabet], dtype=np.float32)


def _build_shared_bias_by_res_dict(shared_bias_by_res, sample_tag):
    return {sample_tag: shared_bias_by_res}

class ProteinMPNN_runner():
    '''
    This class is designed to run the ProteinMPNN model on a single input. This class handles the loading of the model,
    the loading of the input data, the running of the model, and the processing of the output
    '''

    def __init__(self, args, struct_manager):
        self.struct_manager = struct_manager

        if torch.cuda.is_available():
            print('Found GPU will run ProteinMPNN on GPU')
            self.device = "cuda:0"
        else:
            print('No GPU found, running ProteinMPNN on CPU')
            self.device = "cpu"

        self.mpnn_model = mpnn_util.init_seq_optimize_model(
            self.device,
            hidden_dim=128,
            num_layers = 3,
            backbone_noise = args.augment_eps,
            num_connections = args.num_connections,
            checkpoint_path = args.checkpoint_path
        )

        self.temperature = args.temperature
        self.seqs_per_struct = args.seqs_per_struct
        self.omit_AAs = [ letter for letter in args.omit_AAs.upper() if letter in list("ARNDCQEGHILKMFPSTWYVX") ]
        self.allow_x = args.allow_x
        self.bias_AA_dict = _load_jsonl_dict(args.bias_AA_jsonl)
        self.shared_bias_by_res = _load_jsonl_dict(args.bias_by_res_jsonl)

    def sequence_optimize(self, sample_feats: SampleFeatures) -> list[tuple[str, float]]:
        t0 = time.time()

        # Once we have figured out pose I/O without Rosetta this will be easy to swap in
        pdbfile = 'temp.pdb'
        sample_feats.pose.dump_pdb(pdbfile)

        feature_dict = mpnn_util.generate_seqopt_features(pdbfile, sample_feats.chains)
        feature_dict['name'] = sample_feats.tag

        os.remove(pdbfile)

        arg_dict = mpnn_util.set_default_args(self.seqs_per_struct, omit_AAs=self.omit_AAs, allow_x=self.allow_x)
        arg_dict['temperature'] = self.temperature
        if self.bias_AA_dict:
            arg_dict['bias_AA_dict'] = self.bias_AA_dict
            arg_dict['bias_AAs_np'] = _build_bias_AAs_np(self.bias_AA_dict)
        if self.shared_bias_by_res:
            arg_dict['bias_by_res_dict'] = _build_shared_bias_by_res_dict(
                self.shared_bias_by_res,
                sample_feats.tag,
            )

        masked_chains = sample_feats.chains[:-1]
        visible_chains = [sample_feats.chains[-1]]

        fixed_positions_dict = {sample_feats.tag: sample_feats.fixed_res}

        sequences = mpnn_util.generate_sequences(
            self.mpnn_model,
            self.device,
            feature_dict,
            arg_dict,
            masked_chains,
            visible_chains,
            fixed_positions_dict=fixed_positions_dict
        )
        
        print( f"MPNN generated {len(sequences)} sequences in {int( time.time() - t0 )} seconds" ) 

        print(f'sequence_optimize: {sequences}')

        return sequences

    def proteinmpnn(self, sample_feats: SampleFeatures) -> None:
        '''
        Run MPNN sequence optimization on the pose
        '''
        seqs_scores = self.sequence_optimize(sample_feats)

        # Iterate though each seq score pair and thread the sequence onto the pose
        # Then write each pose to a pdb file
        prefix = f"{sample_feats.tag}_dldesign"
        for idx, (seq, score) in enumerate(seqs_scores):
            sample_feats.thread_mpnn_seq(seq)

            outtag = f"{prefix}_{idx}"
            score_str = f"proteinmpnn_score={float(np.asarray(score).item()):.6f}"

            self.struct_manager.dump_pose(sample_feats.pose, outtag, score_str=score_str)

    def run_model(self, tag, args):
        t0 = time.time()

        print(f"Attempting pose: {tag}")
        
        # Load the pose 
        pose = self.struct_manager.load_pose(tag)

        # Initialize the features
        sample_feats = SampleFeatures(pose, tag)

        # Parse the loop string and determine which residues should be designed
        sample_feats.loop_string2fixed_res(args.loop_string)

        self.proteinmpnn(sample_feats)

        seconds = int(time.time() - t0)

        print(f"Struct: {pdb} reported success in {seconds} seconds")


####################
####### Main #######
####################

struct_manager = StructManager(args)
proteinmpnn_runner = ProteinMPNN_runner(args, struct_manager)

for pdb in struct_manager.iterate():

    if args.debug: proteinmpnn_runner.run_model(pdb, args)

    else: # When not in debug mode the script will continue to run even when some poses fail
        t0 = time.time()

        try: proteinmpnn_runner.run_model(pdb, args)

        except KeyboardInterrupt: sys.exit("Script killed by Control+C, exiting")

        except:
            seconds = int(time.time() - t0)
            print(f"Struct with tag {pdb} failed in {seconds} seconds with error: {sys.exc_info()[0]}")

    # We are done with one pdb, record that we finished
    struct_manager.record_checkpoint(pdb)
    
