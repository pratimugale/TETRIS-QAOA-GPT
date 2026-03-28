import os
import time
import json
# import pandas as pd  # Removed to avoid dependency issues
import subprocess
import sys
from datetime import datetime

# --- CONFIGURATION ---
WATCH_DIR = "./out-n12_standard" 
PROJECT_ROOT = "./TETRIS-QAOA-GPT"
CHECK_INTERVAL = 10 
# ---------------------

processed_files = set()

def run_julia_evaluation(json_path):
    """Calls the local Julia environment to evaluate the circuits."""
    print(f"🧬 Starting Julia evaluation for: {json_path}")
    
    # Load metadata and data from the JSON
    try:
        with open(json_path, 'r') as f:
            payload = json.load(f)
    except Exception as e:
        print(f"❌ Error reading {json_path}: {e}")
        return None, None
    
    if not isinstance(payload, dict) or 'meta' not in payload:
        print(f"⚠️ Invalid JSON structure in {json_path} (missing 'meta')")
        return None, None

    meta = payload['meta']
    # Identify if the JSON uses 'data' (v1) or 'circuits' (v2)
    data_key = 'data' if 'data' in payload else 'circuits' if 'circuits' in payload else None
    if not data_key:
        print(f"⚠️ No data key ('data' or 'circuits') found in {json_path}")
        return None, None
        
    raw_data = payload[data_key]
    n_nodes = meta.get('n_nodes', 12)
    pool_type = meta.get('pool_type', 'qaoa_double_pool')
    
    # ---- Robust Cleaning for Julia ----
    def clean_circuit(raw_list):
        if not raw_list or not isinstance(raw_list, list):
            return []
        
        # 1. Julia bridge 'eval_ansatz.jl' expects [Marker (j), Op (j+1), Beta (j+2), Gamma (j+3)]
        # 2. GPT output is [LayerMarker, Op, Gamma, Beta]
        # We must map op_X -> X and swap P1/P2
        
        clean_list = []
        i = 0
        while i < len(raw_list):
            tok = raw_list[i]
            
            # If we find an operator token (op_X or opX)
            if isinstance(tok, str) and (tok.startswith('op_') or (tok.startswith('op') and any(c.isdigit() for c in tok))):
                # Search forward for 2 numeric parameters
                params = []
                k = i + 1
                while k < len(raw_list) and len(params) < 2:
                    val = raw_list[k]
                    if isinstance(val, (int, float)):
                        params.append(float(val))
                    elif isinstance(val, str):
                        try:
                            # Clean up characters like pipes or pads that might be adjacent
                            clean_val = val.strip().replace('|', '').replace('<pad>', '')
                            if clean_val:
                                params.append(float(clean_val))
                        except ValueError:
                            pass
                    k += 1
                
                if len(params) == 2:
                    try:
                        # Extract integer index
                        op_idx = int(''.join(filter(str.isdigit, tok)))
                        # [Marker, Op, Beta, Gamma]
                        # Correct swap: GPT [Op, Gamma, Beta] -> Julia [Op, Beta, Gamma]
                        clean_list.extend([1, op_idx, params[1], params[0]])
                    except:
                        pass
                    i = k - 1 # Skip to end of these parameters
            i += 1
        return clean_list

    cleaned_rows = []
    for item in raw_data:
        # 1. Clean AI predictions
        all_samples_clean = []
        pred_keys = ['q_circuits', 'token_seq_round_d2', 'circuits']
        candidate_circuits = []
        for pk in pred_keys:
            if pk in item and item[pk]:
                candidate_circuits = item[pk]
                break
        
        if candidate_circuits:
            if not isinstance(candidate_circuits[0], list):
                candidate_circuits = [candidate_circuits]
            for raw_list in candidate_circuits:
                all_samples_clean.append(clean_circuit(raw_list))
        
        item['q_circuits'] = all_samples_clean
        
        # 2. Clean ADAPT circuit (Reference)
        if 'adapt_circuit' in item:
            item['adapt_circuit'] = clean_circuit(item['adapt_circuit'])
            
        # 3. Formula preservation
        item['formula_jl'] = item.get('formula_jl') or item.get('formula_list') or []
        cleaned_rows.append(item)

    payload[data_key] = cleaned_rows
    clean_json_path = json_path.replace(".json", "_cleaned.json")
    with open(clean_json_path, 'w') as f:
        json.dump(payload, f)
    # ------------------------------------
    
    # Paths for Julia bridge
    input_file = clean_json_path
    output_file = json_path.replace(".json", "_results.json")
    script_path = os.path.join(PROJECT_ROOT, "adapt_gpt_eval_energy.jl")
    
    cmd = [
        "julia",
        f"--project={PROJECT_ROOT}",
        script_path,
        input_file,
        output_file,
        str(n_nodes),
        pool_type
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Julia evaluation complete. Results saved to: {output_file}")
        if os.path.exists(clean_json_path):
            os.remove(clean_json_path)
        return output_file, meta
    except subprocess.CalledProcessError as e:
        print(f"❌ Julia evaluation failed: {e}")
        return None, None

def update_wandb(results_path, meta):
    """Updates the active WandB run with calculated results."""
    try:
        import wandb
    except ImportError:
        print("⚠️ wandb not installed. Skipping update.")
        return

    run_id = meta.get('wandb_run_id')
    project = meta.get('wandb_project', 'qaoa-gpt-sat-v12-standard')
    iter_num = meta['iter_num']

    if not run_id:
        print("⚠️ No wandb_run_id. Skipping.")
        return

    print(f"📈 Updating WandB run '{run_id}' (Step {iter_num})...")

    with open(results_path, 'r') as f:
        results_data = json.load(f)
    
    # Pure Python replacement for pandas logic
    total_ar = 0.0
    count = 0
    total_samples = 0
    
    for row in results_data:
        energies = row.get('adapt_gpt_energies', [])
        mqlib = row.get('energy_mqlib', 1.0)
        
        if not isinstance(energies, list):
            energies = [energies]
            
        for E in energies:
            total_samples += 1
            if E != 999: # 999 is our error marker
                total_ar += (E / mqlib)
                count += 1
    
    if count == 0:
        print("⚠️ No valid results found.")
        return

    avg_ar = total_ar / count
    error_rate = 1.0 - (count / total_samples) if total_samples > 0 else 1.0

    try:
        run = wandb.init(project=project, id=run_id, resume="must")
        run.log({
            "val/ar": avg_ar,
            "val/er": error_rate,
            "iter": iter_num
        }, step=iter_num)
        run.finish()
        print(f"⭐ WandB Update Successful: AR={avg_ar:.4f}")
    except Exception as e:
        print(f"❌ WandB Error: {e}")

def main():
    search_paths = [".", "../drive"]
    print("👀 Monitoring for circuit predictions...")
    
    while True:
        active_dirs = []
        for p in search_paths:
            if not os.path.exists(p): continue
            for d in os.listdir(p):
                fullpath = os.path.join(p, d)
                if os.path.isdir(fullpath) and (d.startswith("out") or d == "work"):
                    active_dirs.append(fullpath)
        
        for watch_dir in active_dirs:
            target_files = [
                f for f in os.listdir(watch_dir) 
                if f.startswith("eval_circuits_iter_") and f.endswith(".json") 
                and "_results" not in f and "_cleaned" not in f 
                and not f.endswith("_cleaned.json")
            ]
            
            for fname in sorted(target_files):
                fullpath = os.path.normpath(os.path.join(watch_dir, fname))
                if fullpath not in processed_files:
                    print(f"\n🔔 New file detected: {fname}")
                    results_path, meta = run_julia_evaluation(fullpath)
                    if results_path:
                        update_wandb(results_path, meta)
                    processed_files.add(fullpath)
        
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
