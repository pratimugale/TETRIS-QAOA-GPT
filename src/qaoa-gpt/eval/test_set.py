import os
import pickle
import torch
import numpy as np
import pandas as pd
import sys
import json
import ast
from tqdm import tqdm

# Ensure we can import from nanoGPT/ folder
sys.path.append('nanoGPT')
from model_pad_gemb import GPT, GPTConfig

def evaluate_test_set():
    # 1. Configuration & Paths
    checkpoint_path = 'out-adapt_qaoa_5layer_standardqaoamixer/ckpt_600_gemb.pt'
    data_dir = 'src/qaoa-gpt/dataset/adapt_qaoa_5layer_standardqaoamixer'
    device = 'cpu' # Change to 'cuda' or 'mps' if available locally
    
    print("--- Step 1: Loading Metadata and Model ---")
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    itos = meta['itos']

    checkpoint = torch.load(checkpoint_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    # Fix weight names (handle torch.compile prefix)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 2. Loading Test Split
    print("\n--- Step 2: Loading Test Split ---")
    # Note: Using val.npy or test.npy depending on what exists. 
    # In earlier scripts, test_frac was 0 and val_frac was 0.1. 
    # But meta has test_data_formula_idx_list. Let's check test.npy.
    test_npy_path = os.path.join(data_dir, 'test.npy')
    if not os.path.exists(test_npy_path):
        print(f"ERROR: {test_npy_path} not found! Is the split name correct?")
        return

    test_data = np.load(test_npy_path)
    graph_emb_np = np.load(os.path.join(data_dir, 'feather_emb_d500.npy'))
    df = pd.read_csv(os.path.join(data_dir, 'combined_res_tok_shf_df.csv'))

    # Helper to clean circuit tokens for Julia
    def clean_circ(tokens):
        cleaned = []
        for t in tokens:
            t_str = str(t)
            if t_str in ['bos', 'eos', 'pad', 'new_layer_p', 'end_of_formula']:
                if t_str == 'new_layer_p':
                    cleaned.append(t_str)
                continue
            if t_str.startswith('op_'):
                try:
                    cleaned.append(int(t_str.split('_')[1]))
                except:
                    cleaned.append(t_str)
            else:
                try:
                    cleaned.append(float(t_str))
                except:
                    cleaned.append(t_str)
        return cleaned

    idx_list = meta['test_data_formula_idx_list']
    all_samples_for_eval = []

    print(f"\nProcessing all {len(idx_list)} TEST samples...")
    
    for sample_i in tqdm(range(len(idx_list))):
        x_raw = test_data[sample_i, 0]
        eof_idx = np.where(x_raw == stoi['end_of_formula'])[0][0]
        formula_tokens = x_raw[:eof_idx + 1]

        x_input = torch.tensor(formula_tokens.astype(np.int64), dtype=torch.long, device=device).unsqueeze(0)
        
        # Use the ID mapping for robustness
        formula_idx = idx_list[sample_i]
        formula_id = meta['emb_formula_idx_to_id_dict'][formula_idx]
        
        # Extract row from CSV
        matching_rows = df[df['formula_id'] == formula_id]
        if len(matching_rows) == 0:
            continue
        row = matching_rows.iloc[0]
        
        # Generation
        cur_graph_emb = torch.from_numpy(graph_emb_np[formula_idx].astype(np.float32)).to(device).unsqueeze(0)
        with torch.no_grad():
            y_out = model.generate(x_input, cur_graph_emb, max_new_tokens=150, temperature=0.1, top_k=200, eos_id=stoi['eos'])

        generated_only = y_out[0].tolist()[len(formula_tokens):]
        circuit_tokens = [str(itos[t]) for t in generated_only if t != stoi['pad'] and t != stoi['eos']]

        # Extract circuit tokens from CSV
        full_target_seq = ast.literal_eval(row['token_seq_round_d2'])
        gt_eof_idx = full_target_seq.index('end_of_formula')
        gt_circuit_raw = full_target_seq[gt_eof_idx + 1:]

        # Extract Type (Balanced vs Random) from filename
        filename = str(row['filename'])
        formula_type = 'balanced' if 'balanced' in filename.lower() else 'random'

        sample_for_eval = {
            'graph_prefix': f"TEST_{sample_i}__{formula_id}",
            'type': formula_type,
            'formula_jl': json.loads(row['formula_list']),
            'energy_gurobi': float(row['ground_truth_energy']),
            'adapt_circuit': clean_circ(gt_circuit_raw), 
            'q_circuits': [clean_circ(circuit_tokens)]
        }
        all_samples_for_eval.append(sample_for_eval)

    # 4. Save to JSON
    temp_dir = 'tmp_eval'
    os.makedirs(temp_dir, exist_ok=True)
    in_file = os.path.join(temp_dir, 'test_eval_input.json')
    
    with open(in_file, 'w') as f:
        json.dump(all_samples_for_eval, f, indent=4)
    
    print(f"\nSUCCESS! {len(all_samples_for_eval)} TEST samples saved to: {in_file}")
    print("\n--- NEXT STEP: RUN THIS IN YOUR TERMINAL ---")
    print(f"julia -t 4 --project=ADAPT.jl adapt_gpt_eval_energy.jl {in_file} {temp_dir}/test_eval_output.json 12 qaoa_mixer")

if __name__ == "__main__":
    evaluate_test_set()
