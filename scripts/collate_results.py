import os
import csv
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

def collate_results(dataset_metadata_path, results_root, output_path):
    """
    Collates Julia optimization results (JSON) with dataset metadata (CSV).
    
    Args:
        dataset_metadata_path (str): Path to metadata.csv from generation.
        results_root (str): Root directory containing sharded results_json/ folder.
        output_path (str): Path to save the final merged CSV.
    """
    print(f"Loading metadata from {dataset_metadata_path}...")
    df_meta = pd.read_csv(dataset_metadata_path)
    
    # Store indices for fast lookup by filename
    # results will be a list of dicts to be merged
    results_map = {}
    
    results_json_dir = Path(results_root) / "results_json"
    print(f"Scanning for JSON results in {results_json_dir}...")
    
    json_files = list(results_json_dir.rglob("*.json"))
    print(f"Found {len(json_files)} result files.")
    
    for json_path in tqdm(json_files, desc="Parsing JSONs"):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            fname = data.get("instance_filename")
            if not fname:
                print(f"Warning: Missing 'instance_filename' in {json_path}")
                continue

            res_entry = {
                "gamma_coeffs": json.dumps(data.get("gamma_values", [])),
                "beta_coeffs": json.dumps(data.get("beta_values", [])),
                "op_list": json.dumps(data.get("selected_indices", [])),
                "n_layers": data.get("num_adapt_layers", 0),
                "final_energy": data.get("final_energy", 0.0),
                "approx_ratio": data.get("approximation_ratio", 0.0),
                "total_runtime": data.get("total_runtime", 0.0),
                "adapt_runtime": data.get("adapt_runtime", 0.0),
                "gurobi_satisfied": data.get("gurobi_energy", 0.0),
                "initial_gamma": data.get("initial_gamma", 0.0) # Track which gamma was used
            }
            
            # Use list because we expect multiple gammas per filename
            results_map.setdefault(fname, []).append(res_entry)

        except Exception as e:
            print(f"Error parsing {json_path}: {e}")
            
    print("Selecting best results per instance...")
    final_results = {}
    for fname, results_list in results_map.items():
        # Selection logic:
        # 1. Any result with approx_ratio >= 0.999?
        at_goal = [r for r in results_list if r["approx_ratio"] >= 0.999]
        if at_goal:
            # Pick the most compact (min n_layers)
            # If tied, pick highest approx_ratio
            best_res = min(at_goal, key=lambda x: (x["n_layers"], -x["approx_ratio"]))
        else:
            # No result at goal, pick the highest approx_ratio overall
            best_res = max(results_list, key=lambda x: x["approx_ratio"])
        
        final_results[fname] = best_res

    print("Merging results with metadata...")
    
    # Create result columns
    res_cols = ["gamma_coeffs", "beta_coeffs", "op_list", "n_layers", "final_energy", "approx_ratio", "total_runtime", "adapt_runtime", "initial_gamma"]
    for col in res_cols:
        df_meta[col] = df_meta["filename"].map(lambda x: final_results.get(x, {}).get(col))
        
    # Calculate ground_truth_energy (min unsatisfied clauses) using metadata's max_satisfied
    # n_clauses - max_satisfied
    df_meta['ground_truth_energy'] = df_meta['n_clauses'] - df_meta['max_satisfied']
    
    # Drop rows that don't have results (e.g. if Slurm job is partially finished)
    initial_count = len(df_meta)
    df_meta = df_meta.dropna(subset=["op_list"])
    final_count = len(df_meta)
    
    print(f"Successfully merged {final_count} / {initial_count} instances.")
    
    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_meta.to_csv(output_path, index=False)
    print(f"Collated results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collate Julia results into metadata CSV")
    parser.add_argument("--metadata", required=True, help="Path to original metadata.csv")
    parser.add_argument("--results", required=True, help="Root directory of results (containing results_json/)")
    parser.add_argument("--output", required=True, help="Path for the final optimized_circuits.csv")
    
    args = parser.parse_args()
    collate_results(args.metadata, args.results, args.output)
