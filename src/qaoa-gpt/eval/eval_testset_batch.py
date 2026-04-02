"""
eval_testset_batch.py
---------------------
Runs the trained QAOA-GPT model on every instance in the test split,
generates N_SAMPLES circuits per instance, evaluates them with ADAPT.jl,
and reports:
  - mean ± std of Best AR   (best circuit AR per formula)
  - mean ± std of Avg AR    (average circuit AR per formula)
  - mean ± std of Error Rate (fraction of invalid circuits per formula)

AR formula (matching training loop):
  AR = (n_clauses - qaoa_energy) / (n_clauses - gurobi_energy)

Usage:
  python src/qaoa-gpt/eval/eval_testset_batch.py \
      --checkpoint out-n10/ckpt.pt \
      --data_dir   src/qaoa-gpt/dataset/n10 \
      --n_samples  5 \
      --device     cpu
"""

import time
import os
import sys
import json
import pickle
import argparse
import ast
import subprocess
import tempfile

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for HPC

# ── make nanoGPT importable ──────────────────────────────────────────────────
sys.path.append('nanoGPT')

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',  required=True,  help='Path to ckpt.pt')
parser.add_argument('--data_dir',    required=True,  help='Dataset directory (n10, n12 …)')
parser.add_argument('--n_samples',   type=int, default=5, help='Circuits per formula')
parser.add_argument('--device',      default='cpu',  help='cuda / mps / cpu')
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--top_k',       type=int,   default=200)
parser.add_argument('--n_nodes',     type=int,   default=10, help='Qubit count for ADAPT.jl')
parser.add_argument('--pool_type',   default='qaoa_mixer', help='ADAPT pool type')
parser.add_argument('--julia_script', default='src/qaoa-gpt/adapt_gpt_eval_energy.jl')
parser.add_argument('--julia_threads', type=int, default=4)
parser.add_argument('--out_dir',     default='tmp_eval', help='Scratch dir for JSON files')
parser.add_argument('--batch_size',  type=int, default=16, help='Number of formulas per GPU batch')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
device      = args.device
device_type = 'cuda' if 'cuda' in device else 'cpu'

# ── Load meta ─────────────────────────────────────────────────────────────────
print(f"\n[1/5] Loading metadata from {args.data_dir}")
with open(os.path.join(args.data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi = meta['stoi']
itos = meta['itos']
eos_id  = stoi['eos']
pad_id  = stoi['pad']
eof_id  = stoi['end_of_formula']

# ── Load model ────────────────────────────────────────────────────────────────
print(f"[2/5] Loading checkpoint from {args.checkpoint}")
checkpoint  = torch.load(args.checkpoint, map_location=device)
model_args  = checkpoint['model_args']

# Detect whether this is a FEATHER-augmented or baseline model
# The training script saves full config dict with use_graph_emb flag
ckpt_config = checkpoint.get('config', {})
use_gemb    = ckpt_config.get('use_graph_emb', False)
print(f"  Model type: {'FEATHER-augmented (graph emb)' if use_gemb else 'Baseline (no graph emb)'}")

if use_gemb:
    from model_pad_gemb import GPT as GPT_gemb, GPTConfig as GPTConfig_gemb
    gptconf = GPTConfig_gemb(**model_args)
    model   = GPT_gemb(gptconf)
else:
    from model_pad import GPT as GPT_base, GPTConfig as GPTConfig_base
    gptconf = GPTConfig_base(**{k: v for k, v in model_args.items() if k != 'emb_dim'})
    model   = GPT_base(gptconf)

state_dict = checkpoint['model']
prefix = '_orig_mod.'
for k in list(state_dict.keys()):
    if k.startswith(prefix):
        state_dict[k[len(prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

# ── Load test split ───────────────────────────────────────────────────────────
print(f"[3/5] Loading test split …")
test_npy_path = os.path.join(args.data_dir, 'test.npy')
if not os.path.exists(test_npy_path):
    raise FileNotFoundError(f"test.npy not found in {args.data_dir}")

test_data    = np.load(test_npy_path)
graph_emb_np = np.load(os.path.join(args.data_dir, 'feather_emb_d500.npy')) if use_gemb else None

csv_path = os.path.join(args.data_dir, 'combined_res_tok_shf_df.csv')
pkl_path = os.path.join(args.data_dir, 'combined_res_tok_shf_df.pkl')
if os.path.exists(pkl_path):
    df = pd.read_pickle(pkl_path)
elif os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    for col in ['token_seq_round_d2', 'formula_list']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
else:
    raise FileNotFoundError("combined_res_tok_shf_val_df not found")

idx_list              = meta['test_data_formula_idx_list']
emb_idx_to_id         = meta.get('emb_formula_idx_to_id_dict', {})
formula_id_to_emb_idx = meta.get('emb_formula_id_to_idx_dict', {})

# Load optimized_circuits.csv to get adapt_runtime for speedup comparison
opt_circ_path = os.path.join(args.data_dir, 'optimized_circuits.csv')
if os.path.exists(opt_circ_path):
    print(f"  Loading adapt_runtime from {opt_circ_path} …")
    opt_df = pd.read_csv(opt_circ_path)
    # Map filename (formula_id) to adapt_runtime
    runtime_map = opt_df.set_index('filename')['adapt_runtime'].to_dict()
else:
    print(f"  WARNING: {opt_circ_path} not found. Speedup analysis will be skipped.")
    runtime_map = {}

# ── Helper: clean circuit tokens for Julia ────────────────────────────────────
def clean_circ(tokens):
    cleaned = []
    for t in tokens:
        t_str = str(t)
        if t_str in ['bos', 'eos', 'pad', 'end_of_formula']:
            continue
        if t_str == 'new_layer_p':
            cleaned.append(t_str)
        elif t_str.startswith('op_'):
            try:
                cleaned.append(int(t_str.split('_')[1]))
            except Exception:
                cleaned.append(t_str)
        else:
            try:
                cleaned.append(float(t_str))
            except Exception:
                cleaned.append(t_str)
    return cleaned

# ── Generate circuits (Batch Inference Strategy) ───────────────────────────────────
print(f"[4/5] Generating {args.n_samples} circuits for each of {len(idx_list)} test instances (Batch Size: {args.batch_size}) …")

# First pass: Group instances by formula length
len_buckets = {}
for i in range(len(idx_list)):
    x_raw = test_data[i, 0]
    eof_idx = int(np.where(x_raw == eof_id)[0][0])
    formula_tokens = x_raw[:eof_idx + 1] # shape (T,)
    
    L = len(formula_tokens)
    if L not in len_buckets:
        len_buckets[L] = []
    len_buckets[L].append(i)

all_samples = []
ptdtype = torch.bfloat16 if device_type == 'cuda' else torch.float32
from contextlib import nullcontext
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

pbar = tqdm(total=len(idx_list))

# Second pass: Iterate through buckets and process in batches
for L, inst_indices in len_buckets.items():
    # Process formulas in chunks of batch_size
    for i in range(0, len(inst_indices), args.batch_size):
        batch_inst_ids = inst_indices[i : i + args.batch_size]
        B = len(batch_inst_ids)
        
        # Prepare inputs: (B, L)
        formula_batch = []
        embedding_batch = []
        instance_metadata = []
        
        for inst_idx in batch_inst_ids:
            x_raw = test_data[inst_idx, 0]
            formula_tokens = x_raw[:L] 
            formula_batch.append(formula_tokens)
            
            formula_idx = idx_list[inst_idx]
            formula_id  = emb_idx_to_id.get(formula_idx, formula_idx)
            
            # Lookup metadata
            matching_rows = df[df['formula_id'] == formula_id]
            if len(matching_rows) == 0:
                print(f"Warning: No metadata found for formula {formula_id}")
                continue
            row = matching_rows.iloc[0]
            
            # Embeddings
            if use_gemb:
                emb_idx = formula_id_to_emb_idx.get(formula_id, formula_idx)
                embedding_batch.append(graph_emb_np[emb_idx])
            
            instance_metadata.append(row)
            
        if not formula_batch: continue
            
        # Replicate for n_samples: (B * n_samples, L)
        x_input = torch.tensor(np.array(formula_batch), dtype=torch.long, device=device)
        x_input = x_input.repeat_interleave(args.n_samples, dim=0)
        
        if use_gemb:
            cur_graph_emb = torch.tensor(np.array(embedding_batch), dtype=torch.float32, device=device)
            cur_graph_emb = cur_graph_emb.repeat_interleave(args.n_samples, dim=0)
            if device_type == 'cuda':
                cur_graph_emb = cur_graph_emb.to(torch.bfloat16)
        else:
            cur_graph_emb = None

        # Max context 512, generate for remaining space
        max_gen = 512 - L
        
        t_batch_start = time.time()
        with torch.no_grad():
            with ctx:
                if use_gemb:
                    y_out = model.generate(
                        x_input, cur_graph_emb,
                        max_new_tokens=max_gen,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        eos_id=eos_id,
                    )
                else:
                    y_out = model.generate(
                        x_input, 
                        max_gen,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        eos_id=eos_id,
                    )
        t_batch_end = time.time()
        batch_latency = (t_batch_end - t_batch_start) / B
        
        # Post-process: Extract and truncate
        for b_idx in range(B):
            row = instance_metadata[b_idx]
            formula_id = row['formula_id']
            formula_list = row['formula_list']
            if isinstance(formula_list, str):
                formula_list = ast.literal_eval(formula_list)
            
            formula_type = 'balanced' if 'balanced' in str(row.get('filename', '')).lower() else 'random'
            adapt_runtime = runtime_map.get(str(row.get('filename', '')), 0.0)
            
            # Ground truth circuit
            full_target_seq = row['token_seq_round_d2']
            gt_eof_idx      = full_target_seq.index('end_of_formula')
            gt_circuit_raw  = full_target_seq[gt_eof_idx + 1:]
            
            # Generated circuits
            q_circs_batch = []
            for s_idx in range(args.n_samples):
                seq_idx = b_idx * args.n_samples + s_idx
                
                # Truncate at first EOS
                generated_tokens_ids = y_out[seq_idx].tolist()[L:]
                try:
                    eos_index = generated_tokens_ids.index(eos_id)
                    actual_gen_ids = generated_tokens_ids[:eos_index]
                except ValueError:
                    actual_gen_ids = generated_tokens_ids
                
                circuit_tokens = [str(itos[tid]) for tid in actual_gen_ids 
                                 if tid != pad_id and tid != eos_id]
                q_circs_batch.append(clean_circ(circuit_tokens))
                
            all_samples.append({
                'graph_prefix':   f"TEST_{len(all_samples)}__{formula_id}",
                'type':           formula_type,
                'formula_jl':     formula_list,
                'energy_gurobi':  float(row['ground_truth_energy']),
                'adapt_circuit':  clean_circ(gt_circuit_raw),
                'q_circuits':     q_circs_batch,
                'inference_time': batch_latency,
                'adapt_runtime':  adapt_runtime
            })
            
        pbar.update(B)

pbar.close()

# ── Write input JSON ──────────────────────────────────────────────────────────
in_json  = os.path.join(args.out_dir, 'testset_eval_input.json')
out_json = os.path.join(args.out_dir, 'testset_eval_output.json')

with open(in_json, 'w') as f:
    json.dump(all_samples, f, indent=2)
print(f"\nSaved {len(all_samples)} instances to {in_json}")

# ── Call ADAPT.jl ─────────────────────────────────────────────────────────────
print(f"[5/5] Running ADAPT.jl evaluation (threads={args.julia_threads}) …")
julia_cmd = [
    'julia',
    f'-t', str(args.julia_threads),
    '--project=ADAPT.jl',
    args.julia_script,
    in_json,
    out_json,
    str(args.n_nodes),
    args.pool_type,
]
print("  CMD:", ' '.join(julia_cmd))
subprocess.run(julia_cmd, check=True)

# ── Parse output and compute statistics ──────────────────────────────────────
print("\n=== Computing Final Statistics ===")
with open(out_json) as f:
    results = json.load(f)

best_ars, avg_ars, error_rates = [], [], []

for entry in results:
    rq = entry.get('result_quality', {})

    best_ar  = rq.get('best_ar_qaoa_gpt', 0.0)
    avg_ar   = rq.get('avg_ar_qaoa_gpt',  0.0)
    svr      = rq.get('sample_svr',        0.0)
    er       = 1.0 - svr            # Error Rate = 1 - SVR

    best_ars.append(best_ar)
    avg_ars.append(avg_ar)
    error_rates.append(er)

best_ars    = np.array(best_ars)
avg_ars     = np.array(avg_ars)
error_rates = np.array(error_rates)

print(f"\nTotal formulas evaluated: {len(results)}")
print(f"Circuits per formula:     {args.n_samples}")
print()
print(f"{'Metric':<30} {'Mean':>10} {'Std':>10}")
print("-" * 52)
print(f"{'Best AR (per formula)':<30} {best_ars.mean():>10.4f} {best_ars.std():>10.4f}")
print(f"{'Avg  AR (per formula)':<30} {avg_ars.mean():>10.4f} {avg_ars.std():>10.4f}")
print(f"{'Error Rate (per formula)':<30} {error_rates.mean():>10.4f} {error_rates.std():>10.4f}")

# Speedup stats
inference_times = np.array([e.get('inference_time', 0.0) for e in all_samples])
adapt_runtimes  = np.array([e.get('adapt_runtime', 0.0) for e in all_samples])

avg_inf   = inference_times.mean()
avg_adapt = adapt_runtimes.mean()
speedup   = avg_adapt / avg_inf if avg_inf > 0 else 0
print()
print(f"{'Avg Inference Time':<30} {avg_inf:>10.4f}s")
print(f"{'Avg ADAPT Runtime':<30} {avg_adapt:>10.4f}s")
print(f"{'Speedup Factor':<30} {speedup:>10.1f}x")

# Save summary CSV
summary_path = os.path.join(args.out_dir, 'testset_summary.csv')
summary_df   = pd.DataFrame({
    'formula':    [e['graph_prefix']                         for e in results],
    'type':       [e.get('type', 'unknown')                  for e in results],
    'best_ar':    best_ars,
    'avg_ar':     avg_ars,
    'error_rate': error_rates,
    'ar_adapt':   [e['result_quality'].get('ar_qaoa_actual', 0) for e in results],
    'inference_time': inference_times,
    'adapt_runtime':  adapt_runtimes,
})
summary_df.to_csv(summary_path, index=False)
print(f"\nPer-formula results saved to {summary_path}")
print(f"Full Julia output saved to   {out_json}")

