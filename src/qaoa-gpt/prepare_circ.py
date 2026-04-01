# This script:
# 1. Loads the optimized_circuits.csv file
# 2. Tokenizes the formula and the circuit
# 3. Applies sliding window to generate training samples (in the case of max-3-sat, since the max_block_size,
#  that we use is large (512), the sliding window will just create one sample per circuit - this will 
#  completely self contain the input formula and the output circuit.)
# 4. Generates graph embeddings for the formulas
# 5. Saves the training, validation, and test sets

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
import random
import argparse
from joblib import Parallel, delayed
from lcg_feather import SATGraphEmbedder
import ast
import pickle

tqdm.pandas()

parser = argparse.ArgumentParser(description='Prepare SAT datasets for GPT training')
parser.add_argument('--csv_input', type=str, help='Path to consolidated optimized_circuits.csv', required=True)
parser.add_argument('--debug_limit', default=0, type=int, help='Number of input files to sample for speed up (debugging)')
parser.add_argument('--save_dir', type=str, help='Path to save files', required=True)
parser.add_argument('--n_nodes', type=int, help='Number of nodes in the dataset', required=True)
parser.add_argument('--rounding_digits', type=int, default=2, help='Number of digits to round to')
parser.add_argument('--min_block_size', type=int, default=128, help='min sequence length in sliding window') #TODO: do we need this?
parser.add_argument('--max_block_size', type=int, default=512, help='nanoGPT block size')
parser.add_argument('--val_frac', type=float, default=0.1, help='Validation fraction')
parser.add_argument('--test_frac', type=float, default=0.1, help='Test fraction')
parser.add_argument('--max_abs_param_val', type=float, default=10, help='Maximum absolute value of gamma and beta params')
parser.add_argument('--perform_coef_mod_range', type=int, default=True, help='Wrap beta to [0; pi] range; 1 is true (default), 0 is false')
parser.add_argument('--apply_sliding_window', type=bool, default=True, action=argparse.BooleanOptionalAction, help='Apply sliding window to generate training samples')
parser.add_argument('--apply_feather_graph', type=bool, default=True, action=argparse.BooleanOptionalAction, help='Apply feather graph to generate graph embeddings')
parser.add_argument('--n_workers', type=int, default=1, help='Number of workers to use to process ADAPT results')
parser.add_argument('--skip_only_qaoa_circ', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Exclude circuits with only QAOA mixer present')

# Parse the arguments
args = parser.parse_args()

print("Preparing ADAPT circuits for GPT training with the following arguments:")
for arg, value in vars(args).items():
    print(f"\t{arg}: {value}")

save_path_str = args.save_dir
rounding_digits = args.rounding_digits
val_frac = args.val_frac
test_frac = args.test_frac
max_abs_param_val = args.max_abs_param_val
perform_coef_mod_range = bool(args.perform_coef_mod_range)
debug_limit = args.debug_limit
min_block_size = args.min_block_size
max_block_size = args.max_block_size
apply_sliding_window = args.apply_sliding_window
apply_feather_graph = args.apply_feather_graph
n_workers = args.n_workers
n_nodes = args.n_nodes
skip_only_qaoa_circ = args.skip_only_qaoa_circ

if debug_limit:
    print(f'For debugging purposes, limit input results to {debug_limit} files.')

save_path = Path(save_path_str)

# Opening all files

print(f"Loading results directly from CSV: {args.csv_input}")
combined_res_df = pd.read_csv(args.csv_input)

# Pre-processed op_list might need string-to-list conversion
# op_list is a list of lists, containing the operator indices for each tiling (tetris-qaoa) layer
#   it is present as a string in the optimized_circuits.csv (file that is input to prepare_circ.py)
if 'op_list' in combined_res_df.columns:
    if isinstance(combined_res_df['op_list'].iloc[0], str):
        combined_res_df['op_list'] = combined_res_df['op_list'].apply(json.loads)
# gamma coefficients for optimized circuits, stored as a string in optimized_circuits.csv
if 'gamma_coeffs' in combined_res_df.columns:
    combined_res_df['γ_coeff'] = combined_res_df['gamma_coeffs'].apply(json.loads)
# beta coefficients for optimized circuits - this is a list of flattened betas, but correspond 
#   to the order of operators in flattened op_list
# these are also stored as a string in the optimized_circuits.csv
if 'beta_coeffs' in combined_res_df.columns:
    combined_res_df['β_coeff'] = combined_res_df['beta_coeffs'].apply(json.loads)
# formula list that completely describes a 3-CNF formula, stored as a string representation of
#   a list of lists in optimized_circuits.csv. Each inner list is a clause, and each element
#   is a literal (positive or negative integer).
if 'formula_list' in combined_res_df.columns:
    combined_res_df['formula_list'] = combined_res_df['formula_list'].apply(json.loads)

# Map satisfaction metrics for trainer compatibility
combined_res_df['ground_truth_energy'] = combined_res_df["n_clauses"] - combined_res_df['max_satisfied']
# QAOA energy is the final energy of the QAOA circuit found by the classical optimizer
combined_res_df['qaoa_energy'] = combined_res_df['final_energy']
combined_res_df['approx_ratio'] = (combined_res_df['n_clauses'] - combined_res_df['qaoa_energy']) / (combined_res_df['n_clauses'] - combined_res_df['ground_truth_energy'])

# Dummy prefix/id for legacy logic
combined_res_df['prefix'] = 'sat' # not exactly needed but can be kept for compatibility with old code, in case if needed in the future
if 'pooltype' not in combined_res_df.columns:
    # TODO: this can be passed in the arguments of this script instead of hard coding
    combined_res_df['pooltype'] = 'qaoa_mixer'
if 'n_nodes' not in combined_res_df.columns:
    combined_res_df['n_nodes'] = args.n_nodes
if 'n_layers' not in combined_res_df.columns:
    combined_res_df["n_layers"] = combined_res_df['γ_coeff'].apply(len)
else:
    # Ensure it's integer type
    combined_res_df["n_layers"] = combined_res_df["n_layers"].astype(int)

combined_res_df['formula_id'] = (
      combined_res_df['prefix']
    + '_^_'
    + combined_res_df['formula_num'].astype(str)
)

print(combined_res_df['formula_method'].value_counts())

#-----------------------#
# Graph embedding

print("Generating real SAT LCG-FEATHER embeddings (500D)...")
    
# Identify unique formulas by formula_num
unique_formulas_df = combined_res_df.drop_duplicates(subset=['formula_num'])
unique_formula_ids = unique_formulas_df['formula_num'].tolist()
    
# Parse list-of-lists from CSV string representation
formulas_list = [ast.literal_eval(f) if isinstance(f, str) else f for f in unique_formulas_df['formula_list']]
    
# Initialize embedder (using default 500D configurations)
embedder = SATGraphEmbedder(n_nodes=args.n_nodes)
feather_par_emb = embedder.get_embeddings(formulas_list)
    
# Create mapping dictionary for downstream indexing using the string formula_id
emb_formula_id_to_idx_dict = {fid: i for i, fid in enumerate(unique_formulas_df['formula_id'])}

# Ensure that 'has_emb' is correctly populated for every row before the CSV is saved
combined_res_df['has_emb'] = combined_res_df['formula_id'].apply(
    lambda x: x in emb_formula_id_to_idx_dict
)
    
# Set flag for the training config
use_graph_emb = apply_feather_graph

# Create inverse mapping dictionary for meta.pkl
emb_formula_idx_to_id_dict = {i: fid for fid, i in emb_formula_id_to_idx_dict.items()}
#-----------------------#

#-----------------------#

# Filtering 
print("Selecting suitable ansatz...")

combined_res_filt_df = combined_res_df[
    # The betas are handled by julia_mod. The gammas are not modded as they break the circuits 
    # (
    #     combined_res_df['β_coeff'].apply(
    #         lambda x: all([abs(coef) < max_abs_param_val for coef in x])
    #     )
    # )
    # &
    (
        combined_res_df['γ_coeff'].apply(
            lambda x: all([abs(coef) < max_abs_param_val for coef in x])
        )
    )
]
print(f"N circuits after initial coefficient filtering: {len(combined_res_filt_df)}")
print(f"Dropped {len(combined_res_df) - len(combined_res_filt_df)} circuits due to max_abs_param_val > {max_abs_param_val}")

# # We allow standard qaoa circuits for now for baseline testing.
# if skip_only_qaoa_circ:
#     print("Filtering out only QAOA circuits...")
#     n_only_qaoa_circ = combined_res_filt_df['only_qaoa_circ'].sum()
#     print(f"Removing {n_only_qaoa_circ} out of total {len(combined_res_filt_df)}")
#     combined_res_filt_df = combined_res_filt_df[
#         combined_res_filt_df['only_qaoa_circ'] == False
#     ]
    

#-----------------------#
# Tokenization
print("Tokenizing...")
tokens_list = []

## Special symbols
special_symbols_list = [
    'pad',
    'bos',
    'eos',
    'new_layer_p',
    'end_of_formula',
    '|' # Clause separator for SAT
]
tokens_list += special_symbols_list

## SAT Literals
# Note: SAT literals for n_nodes variables
sat_lits = []
for i in range(1, n_nodes + 1):
    sat_lits.append(i)
    sat_lits.append(-i)

sat_tokens = []
for lit in sat_lits:
    if lit > 0:
        sat_tokens.append(f"x{lit}")
    else:
        sat_tokens.append(f"~x{abs(lit)}")
        
print(f"\tTotal tokens for SAT literals: {len(sat_tokens)}")
tokens_list += sat_tokens

## Coeffs

n_steps = int((max_abs_param_val * 2 / (10 ** -rounding_digits) ) + 1)

all_coefs_round_set = set(
    [
        round(coef, rounding_digits) for coef in np.linspace(start=-max_abs_param_val, stop=max_abs_param_val, num=n_steps).tolist()
    ]
)
len(all_coefs_round_set)
tokens_list += list(all_coefs_round_set)

print(f"\tTotal tokens for coefs: {len(all_coefs_round_set)}")

## Operator pool
ops_list = []
for l in combined_res_filt_df['op_list']:
    # TODO: currently this logic only works for standard-QAOA mixer circuits. 
    #  this needs to be further modified for tetris-style circuits by zipping the 
    #  operator indices with the corresponding beta coefficients.
    if isinstance(l[0], list):
        for sub_l in l:
            ops_list += sub_l
    else:
        ops_list += l

ops_list = list(set([f"op_{op}" for op in ops_list]))
print(f"\tTotal tokens for operator pool: {len(ops_list)}")
tokens_list += ops_list

## Tokenization
int_idx_to_token_dict = dict(enumerate(tokens_list))
token_to_int_idx_dict = {v:k for k,v in int_idx_to_token_dict.items()}

vocab_size = len(int_idx_to_token_dict)
print(f"\tTotal tokens in vocab: {vocab_size}")

def julia_mod(a, b):
    result = a % b
    return result if a >= 0 else result - b

def tokenize_row(row, coef_mod=True):

    tokens_seq_list = ['bos']

    # SAT formula tokenization
    for clause in row['formula_list']:
        for lit in clause:
            if lit > 0:
                tokens_seq_list.append(f"x{lit}")
            else:
                tokens_seq_list.append(f"~x{abs(lit)}")
        tokens_seq_list.append('|')
    tokens_seq_list.append('end_of_formula')

    for p in range(row['n_layers']):
        tokens_seq_list.append('new_layer_p')
        # TODO: currently this logic only works for standard-QAOA mixer circuits. 
        #  this needs to be further modified for tetris-style circuits by zipping the 
        #  operator indices with the corresponding beta coefficients.
        op_val = row['op_list'][p]
        if isinstance(op_val, list):
            op_val = op_val[0] 
        tokens_seq_list.append(f"op_{op_val}")

        cur_beta = row['β_coeff'][p]
        if coef_mod:
            cur_beta = julia_mod(cur_beta, np.pi)
        if cur_beta > -max_abs_param_val and cur_beta < max_abs_param_val:
            cur_beta_round = round(cur_beta, rounding_digits)
            tokens_seq_list.append(cur_beta_round)
        else:
            return None

        cur_gamma = row['γ_coeff'][p]
        if cur_gamma > -max_abs_param_val and cur_gamma < max_abs_param_val:
            cur_gamma_round = round(cur_gamma, rounding_digits)
            tokens_seq_list.append(cur_gamma_round)
        else:
            return None
    
    tokens_seq_list.append('eos')
    
    return tokens_seq_list

combined_res_filt_df[f'token_seq_round_d{rounding_digits}'] = combined_res_filt_df.progress_apply(
    lambda x: tokenize_row(x, coef_mod=perform_coef_mod_range),
    axis=1,
)
combined_res_tok_df = combined_res_filt_df.dropna()
n_dropped_tok = len(combined_res_filt_df) - len(combined_res_tok_df)
print(f"N circuits after tokenization dropna: {len(combined_res_tok_df)}")
if n_dropped_tok > 0:
    print(f"Dropped {n_dropped_tok} circuits during tokenization (likely out of range or NaN)")
combined_res_tok_df[f'token_int_seq_round_d{rounding_digits}'] = (
    combined_res_tok_df[f'token_seq_round_d{rounding_digits}'].progress_apply(
        lambda x: [token_to_int_idx_dict[token] for token in x]
    )
)

# Generating training split for nanoGPT

print("Preparing training data...")

n = len(combined_res_tok_df)

combined_res_tok_shf_df = (
    combined_res_tok_df
        .sample(frac=1)
        .reset_index(drop=True)
)

print(f"combined_res_df shape: {combined_res_df.shape}")
print(f"combined_res_tok_df shape: {combined_res_tok_df.shape}")
print(f"combined_res_tok_shf_df shape: {combined_res_tok_shf_df.shape}")

# Stratified split based on is_sat and formula_method
from sklearn.model_selection import train_test_split
    
# Create a stratification key
combined_res_tok_shf_df['stratify_key'] = combined_res_tok_shf_df['is_sat'].astype(str) + "_" + combined_res_tok_shf_df['formula_method']
    
train_idx, val_test_idx = train_test_split(
    combined_res_tok_shf_df.index,
    test_size=(val_frac + test_frac),
    stratify=combined_res_tok_shf_df['stratify_key'],
    random_state=42
)
    
val_idx, test_idx = train_test_split(
    val_test_idx, 
    test_size=(test_frac / (val_frac + test_frac)), 
    stratify=combined_res_tok_shf_df.loc[val_test_idx, 'stratify_key'],
    random_state=42
)
    
train_formula_ids_set = set(combined_res_tok_shf_df.loc[train_idx, 'formula_id'])
val_formula_ids_set = set(combined_res_tok_shf_df.loc[val_idx, 'formula_id'])
test_formula_ids_set = set(combined_res_tok_shf_df.loc[test_idx, 'formula_id'])

assert len(train_formula_ids_set.intersection(val_formula_ids_set)) == 0
assert len(train_formula_ids_set.intersection(test_formula_ids_set)) == 0
assert len(val_formula_ids_set.intersection(test_formula_ids_set)) == 0

def pad_with_zeros(seq, target_len):
    # Ensure it's exactly target_len
    if len(seq) > target_len:
        seq = seq[:target_len]
    pad_len = target_len - len(seq)
    if pad_len > 0:
        padded_seq = seq + [0]*pad_len
    else:
        padded_seq = seq
        
    if len(padded_seq) !=max_block_size:
        print(f"padded_seq len: {len(padded_seq)}")
    return padded_seq

def sliding_window(numbers, min_block_size, max_block_size):
    # NO-SLIDE Logic: We only take the very first window starting at Index 0 (bos)
    # X is the first max_block_size tokens
    # Y is the first max_block_size tokens, shifted by 1
    x = numbers[:max_block_size]
    y = numbers[1:max_block_size+1]
    
    return [
        [
            pad_with_zeros(x, target_len=max_block_size),
            pad_with_zeros(y, target_len=max_block_size)
        ]
    ]


# Assign the 'label' column based on the split
combined_res_tok_shf_df['label'] = 'train'
combined_res_tok_shf_df.loc[combined_res_tok_shf_df['formula_id'].isin(val_formula_ids_set), 'label'] = 'val'
combined_res_tok_shf_df.loc[combined_res_tok_shf_df['formula_id'].isin(test_formula_ids_set), 'label'] = 'test'

if apply_sliding_window:
    print('Applying sliding window...')
    combined_res_tok_shf_df[f'token_int_seq_round_d{rounding_digits}_sw'] = combined_res_tok_shf_df[f'token_int_seq_round_d{rounding_digits}'].progress_apply(
        lambda x: sliding_window(
            x,
            min_block_size=min_block_size,
            max_block_size=max_block_size
        )
    )
    
train_data = combined_res_tok_shf_df[
    combined_res_tok_shf_df['label'] == 'train'
]
val_data = combined_res_tok_shf_df[
    combined_res_tok_shf_df['label'] == 'val'
]
test_data = combined_res_tok_shf_df[
    combined_res_tok_shf_df['label'] == 'test'
]

print(f"\tNumber of training samples: {len(train_data)}, val samples: {len(val_data)}, test samples: {len(test_data)}")

if apply_sliding_window:

    train_data_conc = []
    train_data_formula_idx_list = []
    for cur_formula_id, l in zip(
        train_data['formula_id'],
        train_data[f'token_int_seq_round_d{rounding_digits}_sw']
    ):
        if cur_formula_id in emb_formula_id_to_idx_dict:
            train_data_conc += l
            train_data_formula_idx_list += [emb_formula_id_to_idx_dict[cur_formula_id]] * len(l)
    train_data_conc_np = np.array(train_data_conc, dtype=np.uint16)

    val_data_conc = []
    val_data_formula_idx_list = []
    for cur_formula_id, l in zip(
        val_data['formula_id'],
        val_data[f'token_int_seq_round_d{rounding_digits}_sw']
    ):
        if cur_formula_id in emb_formula_id_to_idx_dict:
            val_data_conc += l
            val_data_formula_idx_list += [emb_formula_id_to_idx_dict[cur_formula_id]] * len(l)
    val_data_conc_np = np.array(val_data_conc, dtype=np.uint16)

    test_data_conc = []
    test_data_formula_idx_list = []
    for cur_formula_id, l in zip(
        test_data['formula_id'],
        test_data[f'token_int_seq_round_d{rounding_digits}_sw']
    ):
        if cur_formula_id in emb_formula_id_to_idx_dict:
            test_data_conc += l
            test_data_formula_idx_list += [emb_formula_id_to_idx_dict[cur_formula_id]] * len(l)
    test_data_conc_np = np.array(test_data_conc, dtype=np.uint16)


    # print(f"\tTrain has {len(train_data_conc_np):,} samples")
    # print(f"\tVal has {len(val_data_conc_np):,} samples")
    # print(f"\tTest has {len(test_data_conc_np):,} samples")

# Saving

save_path.mkdir(parents=True, exist_ok=True)

if apply_sliding_window:
    
    np.save(
        save_path.joinpath('train.npy'),
        train_data_conc_np
    )
    np.save(
        save_path.joinpath('val.npy'),
        val_data_conc_np
    )
    np.save(
        save_path.joinpath('test.npy'),
        test_data_conc_np
    )

combined_res_df.to_pickle(
    save_path.joinpath('combined_res_df.pkl')
)
combined_res_df.to_csv(
    save_path.joinpath('combined_res_df.csv'), index=False
)

combined_res_tok_shf_df.to_pickle(
    save_path.joinpath('combined_res_tok_shf_df.pkl')
)
combined_res_tok_shf_df.to_csv(
    save_path.joinpath('combined_res_tok_shf_df.csv'), index=False
)

target_val_size = min(1000, len(val_data))
# simple sampling for SAT
val_data_sampled = val_data.sample(min(len(val_data), target_val_size)).reset_index(drop=True)

val_data_sampled.to_pickle(
    save_path.joinpath('combined_res_tok_shf_val_df.pkl')
)
val_data_sampled.to_csv(
    save_path.joinpath('combined_res_tok_shf_val_df.csv'), index=False
)

emb_size = feather_par_emb.shape[1]
np.save(
    save_path.joinpath(f'feather_emb_d{emb_size}.npy'),
    feather_par_emb
)

meta = {
    'vocab_size': vocab_size,
    'itos': int_idx_to_token_dict,
    'stoi': token_to_int_idx_dict,
    'train_data_formula_idx_list': train_data_formula_idx_list,
    'val_data_formula_idx_list': val_data_formula_idx_list,
    'test_data_formula_idx_list': test_data_formula_idx_list,
    'emb_formula_id_to_idx_dict': emb_formula_id_to_idx_dict,
    'emb_formula_idx_to_id_dict': emb_formula_idx_to_id_dict,
}

with open(save_path.joinpath('meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

script_dir = Path(__file__).parent
with open(script_dir.joinpath('train_adapt_gpt_config_template.py')) as f:
    config_template_str = f.read()

if 'pooltype' in combined_res_df.columns:
    pool_type = combined_res_df['pooltype'].iloc[0]
else:
    pool_type = 'qaoa_mixer'

dataset_name = save_path.stem
config_to_save_str = config_template_str.format(
    out_dir=f'out-{dataset_name}',
    dataset=dataset_name,
    block_size=max_block_size,
    use_graph_emb=use_graph_emb,
    eval_ar_locally="True", # Default to True for Colab-bound configs
    pool_type=pool_type,
    n_nodes=n_nodes,
    rounding_digits=rounding_digits,
)

with open(save_path.joinpath('train_adapt_gpt_config.py'), 'w') as f:
    f.write(config_to_save_str)

print(f"Data is saved to: {str(save_path.absolute())}")
print("Done!")

