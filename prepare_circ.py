import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import networkx as nx
import numpy as np
from collections import Counter
import random
import argparse

tqdm.pandas()

# rounding_digits = 2
# val_frac = 0.1
# approx_ratio_thr = 0.98
# max_abs_param_val = 10

parser = argparse.ArgumentParser(description='Parser for ADAPT GPT circuit preparation.')
parser.add_argument('--adapt_results_dir', type=str, help='Path to read results from')
parser.add_argument('--save_dir', type=str, help='Path to save files')
parser.add_argument('--rounding_digits', type=int, default=2, help='Number of digits to round to')
parser.add_argument('--val_frac', type=float, default=0.1, help='Validation fraction')
parser.add_argument('--test_frac', type=float, default=0.1, help='Test fraction')
parser.add_argument('--approx_ratio_thr', type=float, default=0.98, help='Approximation ratio threshold')
parser.add_argument('--max_abs_param_val', type=float, default=10, help='Maximum absolute value of gamma and betta params')

# Parse the arguments
args = parser.parse_args()

results_fpath_str = args.adapt_results_dir
save_path_str = args.save_dir
rounding_digits = args.rounding_digits
val_frac = args.val_frac
test_frac = args.test_frac
approx_ratio_thr = args.approx_ratio_thr
max_abs_param_val = args.max_abs_param_val

results_fpath = Path(results_fpath_str)
save_path = Path(save_path_str)

assert results_fpath.exists() and results_fpath.is_dir(), "Results path is invalid."

# Opening all files

## ADAPT results
results_folders_list = [results_fpath]
print("Reading ADAPT.jl results from:")
for res_fpath in results_folders_list:
    print('\t', str(res_fpath.absolute()))
    
df_list = []
for cur_dataset_res_path in results_folders_list:
    cur_dataset_res_flist = list(cur_dataset_res_path.joinpath('res').glob('*.csv'))
    for fname in tqdm(cur_dataset_res_flist, desc='Opening ADAPT results'):
        cur_df = pd.read_csv(fname)
        cur_df['worker_id'] = fname.stem
        df_list.append(cur_df)

full_run_df = pd.concat(df_list)
full_run_df['Layer_p'] = (
    full_run_df.groupby(['graph_num', "run", "worker_id"])
        .cumcount()
)
full_run_df['prefix'] = full_run_df['worker_id'].apply(
    lambda x: x[:-15]
)

## Graphs
df_list = []
for cur_dataset_res_path in results_folders_list:
    cur_dataset_res_flist = list(cur_dataset_res_path.joinpath('graphs').glob('*.csv'))
    for fname in tqdm(cur_dataset_res_flist, desc='Opening graphs'):
        cur_df = pd.read_csv(fname)
        cur_df['worker_id'] = fname.stem
        df_list.append(cur_df)
full_run_graphs_df = pd.concat(df_list)
full_run_graphs_df['edgelist_list'] = (
    full_run_graphs_df['edgelist_json'].progress_apply(
        lambda x: json.loads(x)
    )
)
full_run_graphs_df['edgelist_list_len'] = (
    full_run_graphs_df['edgelist_list'].progress_apply(
        lambda x: len(x)
    )
)
full_run_graphs_df['num_connected_comp'] = full_run_graphs_df['edgelist_list'].progress_apply(
    lambda x: len(
        list(
            nx.connected_components(
                nx.Graph([edge[:2] for edge in x])
            )
        )
    )
)
full_run_graphs_df['prefix'] = full_run_graphs_df['worker_id'].apply(
    lambda x: x[:-12]
)

# Aggregating results
print("Aggregating results...")

full_run_df['approx_ratio'] = (
    full_run_df['energy'] / full_run_df['energy_mqlib']
)

full_run_agg_stat_df = (
    full_run_df.groupby(
        ["graph_num", "run", "prefix"]
    )
    .agg(
        {
            'pooltype': 'count',
            'energy': list,
            'took_time': 'last',
            'energy_mqlib': 'last',
            'generator_index_in_pool': list,
            'approx_ratio': 'last',
            'β_coeff': list,
            'γ_coeff': list,
            'energy_mqlib': 'last'
        }
    )
    .reset_index()
    .rename(
        columns={
            'pooltype': 'n_layers',
            'energy': 'energy_list',
            #'took_time': 'took_time',
            'generator_index_in_pool': 'op_list'
        }
    )
)

combined_res_df = pd.merge(
    left=full_run_agg_stat_df,
    right=full_run_graphs_df,
    left_on=['prefix', 'graph_num'],
    right_on=['prefix', 'graph_num'],
)

combined_res_df['graph_id'] = (
      combined_res_df['prefix']
    + '_^_'
    + combined_res_df['graph_num'].astype(str)
)

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
    &
    (
        combined_res_df['approx_ratio'] > approx_ratio_thr
    )
]

# Tokenization
print("Tokenizing...")
tokens_list = []

## Special symbols
special_symbols_list = [
    'bos',
    'eos',
    'new_layer_p',
    'end_of_graph'
]
tokens_list += special_symbols_list

## Edges
all_edges_list = []
for g in combined_res_filt_df['edgelist_list']:
    for e in g:
        all_edges_list.append(tuple(e[:2]))
all_edges_set = set(all_edges_list)

print(f"\tTotal tokens for edges: {len(all_edges_set)}")
tokens_list += list(all_edges_set)

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
for l in combined_res_df['op_list']:
    ops_list += l

ops_list = list(set(ops_list))
print(f"\tTotal tokens for operator pool: {len(ops_list)}")
tokens_list += ops_list

## Tokenization
int_idx_to_token_dict = dict(enumerate(tokens_list))
token_to_int_idx_dict = {v:k for k,v in int_idx_to_token_dict.items()}

vocab_size = len(int_idx_to_token_dict)
print(f"\tTotal tokens in vocab: {vocab_size}")

def tokenize_row(row):

    tokens_seq_list = ['bos']

    for edge in row['edgelist_list']:
        edge_tuple = tuple(edge[:2])
        edge_weight = edge[2]
        tokens_seq_list.append(edge_tuple)
        tokens_seq_list.append(edge_weight)

    tokens_seq_list.append('end_of_graph')

    for p in range(row['n_layers']):
        tokens_seq_list.append('new_layer_p')
        tokens_seq_list.append(row['op_list'][p])

        cur_beta = row['β_coeff'][p]
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

combined_res_df[f'token_seq_round_d{rounding_digits}'] = combined_res_df.progress_apply(
    tokenize_row,
    axis=1
)
combined_res_tok_df = combined_res_df.dropna()
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

graph_ids = combined_res_tok_shf_df['graph_id'].drop_duplicates().to_list()

# Compute the number of graphs for each set
n_total = len(graph_ids)
n_test = int(n_total * test_frac)  # Define test_frac for the size of the test set
n_val = int(n_total * val_frac)  # val_frac defines the validation set size
n_train = n_total - n_test - n_val  # Remaining will be the training set

# Split into train, val, and test sets
train_graph_ids_set = set(graph_ids[:n_train])
val_graph_ids_set = set(graph_ids[n_train:n_train + n_val])
test_graph_ids_set = set(graph_ids[n_train + n_val:])

assert len(train_graph_ids_set.intersection(val_graph_ids_set)) == 0
assert len(train_graph_ids_set.intersection(test_graph_ids_set)) == 0
assert len(val_graph_ids_set.intersection(test_graph_ids_set)) == 0

# Assign the 'label' column based on the split
combined_res_tok_shf_df['label'] = 'train'
combined_res_tok_shf_df.loc[combined_res_tok_shf_df['graph_id'].isin(val_graph_ids_set), 'label'] = 'val'
combined_res_tok_shf_df.loc[combined_res_tok_shf_df['graph_id'].isin(test_graph_ids_set), 'label'] = 'test'

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

train_data_conc = []
for l in train_data[f'token_int_seq_round_d{rounding_digits}']:
    train_data_conc += l
train_data_conc_np = np.array(train_data_conc, dtype=np.uint16)

val_data_conc = []
for l in val_data[f'token_int_seq_round_d{rounding_digits}']:
    val_data_conc += l
val_data_conc_np = np.array(val_data_conc, dtype=np.uint16)

test_data_conc = []
for l in test_data[f'token_int_seq_round_d{rounding_digits}']:
    test_data_conc += l
test_data_conc_np = np.array(test_data_conc, dtype=np.uint16)


print(f"\tTrain has {len(train_data_conc_np):,} tokens")
print(f"\tVal has {len(val_data_conc_np):,} tokens")
print(f"\tTest has {len(test_data_conc_np):,} tokens")

# Saving
save_path.mkdir(parents=True, exist_ok=True)

combined_res_df.to_pickle(
    save_path.joinpath('combined_res_df.pkl')
)

combined_res_tok_shf_df.to_pickle(
    save_path.joinpath('combined_res_tok_shf_df.pkl')
)

train_data_conc_np.tofile(save_path.joinpath('train.bin'))
val_data_conc_np.tofile(save_path.joinpath('val.bin'))
test_data_conc_np.tofile(save_path.joinpath('test.bin'))

meta = {
    'vocab_size': vocab_size,
    'itos': int_idx_to_token_dict,
    'stoi': token_to_int_idx_dict,
}

pd.to_pickle(
    meta,
    save_path.joinpath('meta.pkl')
)

with open('train_adapt_gpt_config_template.py') as f:
    config_template_str = f.read()

dataset_name = save_path.stem
config_to_save_str = config_template_str.format(
    out_dir=f'out-{dataset_name}',
    dataset=dataset_name
)

with open(save_path.joinpath('train_adapt_gpt_config.py'), 'w') as f:
    f.write(config_to_save_str)

print(f"Data is saved to: {str(save_path.absolute())}")
print("Done!")

