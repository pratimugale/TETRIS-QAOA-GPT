import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import networkx as nx
import numpy as np
from collections import Counter
import random
import argparse
from joblib import Parallel, delayed
from FEATHER.src.feather import FEATHERG
from itertools import islice

tqdm.pandas()

# rounding_digits = 2
# val_frac = 0.1
# approx_ratio_thr = 0.98
# max_abs_param_val = 10
parser = argparse.ArgumentParser(description='Prepare SAT datasets for GPT training')
parser.add_argument('--csv_input', type=str, help='Path to consolidated optimized_circuits.csv', required=True)
parser.add_argument('--debug_limit', default=0, type=int, help='Number of input files to sample for speed up (debugging)')
parser.add_argument('--save_dir', type=str, help='Path to save files', required=True)
parser.add_argument('--n_nodes', type=int, help='Number of nodes in the dataset', required=True)
parser.add_argument('--rounding_digits', type=int, default=2, help='Number of digits to round to')
parser.add_argument('--min_block_size', type=int, default=128, help='min sequence length in sliding window')
parser.add_argument('--max_block_size', type=int, default=256, help='nanoGPT block size')
parser.add_argument('--val_frac', type=float, default=0.1, help='Validation fraction')
parser.add_argument('--test_frac', type=float, default=0.1, help='Test fraction')
parser.add_argument('--approx_ratio_thr', type=float, default=0.97, help='Approximation ratio threshold')
parser.add_argument('--max_abs_param_val', type=float, default=10, help='Maximum absolute value of gamma and beta params')
parser.add_argument('--perform_coef_mod_range', type=int, default=True, help='Wrap beta to [0; pi] range; 1 is true (default), 0 is false')
parser.add_argument('--apply_sliding_window', type=bool, default=True, action=argparse.BooleanOptionalAction, help='Apply sliding window to generate training samples')
parser.add_argument('--apply_feather_graph', type=bool, default=True, action=argparse.BooleanOptionalAction, help='Apply feather graph to generate graph embeddings')
parser.add_argument('--n_workers', type=int, default=1, help='Number of workers to use to process ADAPT results')
parser.add_argument('--skip_only_qaoa_circ', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Exclude circuits with only QAOA mixer present')
parser.add_argument('--allowed_formula_type', type=str, default="all", help='Allowed graph generators. Default: all. Should be separated with ;. Allowed values: erdos_renyi;barabasi_albert;watts_strogatz;random_regular;bipartite')
parser.add_argument('--is_sat_mode', type=bool, default=True, action=argparse.BooleanOptionalAction, help='Enable SAT mode (disables graph embeddings)')

# Parse the arguments
args = parser.parse_args()

print("Preparing ADAPT circuits for GPT training with the following arguments:")
for arg, value in vars(args).items():
    print(f"\t{arg}: {value}")

# results_fpath_str = args.adapt_results_dir # This was a bug; adapt_results_dir is not used in SAT mode.
save_path_str = args.save_dir
rounding_digits = args.rounding_digits
val_frac = args.val_frac
test_frac = args.test_frac
approx_ratio_thr = args.approx_ratio_thr
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
allowed_formula_type_list = args.allowed_formula_type.split(';')
is_sat_mode = args.is_sat_mode

if debug_limit:
    print(f'For debugging purposes, limit input results to {debug_limit} files.')

save_path = Path(save_path_str)

# Opening all files

print(f"Loading results directly from CSV: {args.csv_input}")
combined_res_df = pd.read_csv(args.csv_input)

# Pre-processed op_list might need string-to-list conversion
if 'op_list' in combined_res_df.columns:
    if isinstance(combined_res_df['op_list'].iloc[0], str):
        combined_res_df['op_list'] = combined_res_df['op_list'].apply(json.loads)
if 'gamma_coeffs' in combined_res_df.columns:
    combined_res_df['γ_coeff'] = combined_res_df['gamma_coeffs'].apply(json.loads)
if 'beta_coeffs' in combined_res_df.columns:
    combined_res_df['β_coeff'] = combined_res_df['beta_coeffs'].apply(json.loads)
if 'formula_list' in combined_res_df.columns:
    combined_res_df['formula_list'] = combined_res_df['formula_list'].apply(json.loads)

# Map satisfaction metrics for trainer compatibility
combined_res_df['ground_truth_energy'] = combined_res_df["n_clauses"] - combined_res_df['max_satisfied']
combined_res_df['qaoa_energy'] = combined_res_df['final_energy']
combined_res_df['approx_ratio'] = (combined_res_df['n_clauses'] - combined_res_df['qaoa_energy']) / (combined_res_df['n_clauses'] - combined_res_df['ground_truth_energy'])

# Dummy prefix/id for legacy logic
combined_res_df['prefix'] = 'sat' # not exactly needed but can be kept for compatibility with old code
if 'pooltype' not in combined_res_df.columns:
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


if allowed_formula_type_list != ['all']:
    print(f"Filtering graphs based on allowed generators: {allowed_formula_type_list}")
    print(f"N circuits before: {len(combined_res_df)}")
    combined_res_df = combined_res_df[
        combined_res_df['formula_method'].isin(allowed_formula_type_list)
    ]
    print(f"N circuits after: {len(combined_res_df)}")

print(combined_res_df['formula_method'].value_counts())

#-----------------------#
# Graph embedding (SAT Mode)
# TODO: Placeholder for SAT Logical Circuit Graph (LCG) embedding or similar 
# representations to be added here for SAT datasets if needed in the future.
# For now, we use a pass-through (dummy) embedding.
emb_formula_id_to_idx_dict = {gid: i for i, gid in enumerate(combined_res_df['formula_id'].unique())}
emb_formula_idx_to_id_dict = {i: gid for gid, i in emb_formula_id_to_idx_dict.items()}
feather_par_emb = np.zeros((len(emb_formula_id_to_idx_dict), 1))
combined_res_df['has_emb'] = True
#-----------------------#

#-----------------------#

# Filtering 
print("Selecting suitable ansatz...")

combined_res_filt_df = combined_res_df[
    (
        combined_res_df['β_coeff'].apply(
            lambda x: all([abs(coef) < max_abs_param_val for coef in x])
        )
    )
    &
    (
        combined_res_df['γ_coeff'].apply(
            lambda x: all([abs(coef) < max_abs_param_val for coef in x])
        )
    )
]

# # We allow standard qaoa circuits for now for baseline testing.
# if skip_only_qaoa_circ:
#     print("Filtering out only QAOA circuits...")
#     n_only_qaoa_circ = combined_res_filt_df['only_qaoa_circ'].sum()
#     print(f"Removing {n_only_qaoa_circ} out of total {len(combined_res_filt_df)}")
#     combined_res_filt_df = combined_res_filt_df[
#         combined_res_filt_df['only_qaoa_circ'] == False
#     ]
    

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
    # Handle nested lists if present
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
        # Handle nested op list
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
    pad_len = target_len - len(seq)
    if pad_len > 0:
        padded_seq = seq + [0]*pad_len
    else:
        padded_seq = seq
        
    if len(padded_seq) !=max_block_size:
        print(f"padded_seq len: {len(padded_seq)}")
    return padded_seq

def sliding_window(numbers, min_block_size, max_block_size):
    
    if min_block_size != max_block_size:
        block_size = random.randint(min_block_size, max_block_size)
    else:
        block_size = min_block_size

    if block_size >= len(numbers):
        window = numbers[:-1]
        window_shifted = numbers[1:]   
        return [
            [
                pad_with_zeros(window, target_len=max_block_size),
                pad_with_zeros(window_shifted, target_len=max_block_size)
            ]
        ]
    
    result_xy_list = []
    result = []
    for i in range(0, len(numbers) - block_size + 1):
        window = numbers[i:i + block_size]
        result.append(window)
        
    for x, y in zip(result, result[1:]):
        result_xy_list.append(
            [
                pad_with_zeros(x, target_len=max_block_size),
                pad_with_zeros(y, target_len=max_block_size)
            ]
        )
    
    return result_xy_list


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

combined_res_tok_shf_df.to_pickle(
    save_path.joinpath('combined_res_tok_shf_df.pkl')
)

target_val_size = min(1000, len(val_data))
# simple sampling for SAT
val_data_sampled = val_data.sample(min(len(val_data), target_val_size)).reset_index(drop=True)

val_data_sampled.to_pickle(
    save_path.joinpath('combined_res_tok_shf_val_df.pkl')
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

pd.to_pickle(
    meta,
    save_path.joinpath('meta.pkl')
)

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
    use_graph_emb="False" if is_sat_mode else "True",
    pool_type=pool_type,
    n_nodes=n_nodes,
)

with open(save_path.joinpath('train_adapt_gpt_config.py'), 'w') as f:
    f.write(config_to_save_str)

print(f"Data is saved to: {str(save_path.absolute())}")
print("Done!")

