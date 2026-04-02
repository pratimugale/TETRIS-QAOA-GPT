import json
import sys
import os
import subprocess

# Add current directory to path
sys.path.append(os.getcwd())

def run_reproduce_test():
    n_nodes = 10
    # The source data uses the non-diagonal pool to allow more flexibility in operator selection
    pool_type = "qaoa_nondiagonal_double_pool" 
    target_energy = 0.973298316266657
    
    # Exact formula for random_v10_c51_fd1943f0.cnf
    formula_jl = [[1, 2, -3], [1, 2, -9], [1, -2, 4], [1, -3, 5], [1, -3, -6], [1, -4, 6], [1, -4, -6], [1, 5, 6], [1, 5, 10], [1, -5, -9], [1, 7, 9], [1, -9, -10], [-1, 2, 3], [-1, -2, 5], [-1, 3, -4], [-1, -3, 8], [-1, -4, -5], [-1, -4, 9], [-1, -7, 10], [2, 3, 6], [2, 3, -9], [2, -4, -9], [2, -5, 9], [2, -6, -8], [2, 7, -10], [2, -8, 9], [2, -9, -10], [-2, 4, 8], [-2, -4, -9], [-2, -5, -7], [3, 4, 6], [3, -4, -10], [3, -5, 7], [-3, 4, 5], [-3, -6, -7], [4, 5, -7], [4, -6, 10], [4, 9, 10], [-4, 5, 8], [-4, -5, 8], [-4, 6, 7], [-4, -6, -8], [-4, 8, -10], [-5, -6, -9], [-5, -7, -9], [-5, -7, 10], [6, 7, 8], [6, 9, 10], [-6, 7, -10], [7, 8, 10], [7, -8, -9]]
    
    # Extract data from src/qaoa-gpt/tests/tmp/tetris-circuit.json
    with open("src/qaoa-gpt/tests/tmp/tetris-circuit.json", "r") as f:
        data = json.load(f)

    
        
    indices = data["selected_indices"] # List of lists
    gammas = data["gamma_values"]
    betas = data["beta_values"]
    
    # Construct the q_circuit sequence in the new token format
    # Each layer: new_layer_p, op1, beta1, op2, beta2, ..., gamma
    q_circuit = []
    beta_idx = 0
    
    # We take the first 5 layers (as there are 5 gammas and 49 betas)
    for layer_num in range(5):
        q_circuit.append("new_layer_p")
        layer_ops = indices[layer_num]
        for op in layer_ops:
            q_circuit.append(op)
            q_circuit.append(betas[beta_idx])
            beta_idx += 1
        q_circuit.append(gammas[layer_num])
    
    print(f"Reproducing exact Tetris-QAOA energy for {data['instance_filename']}...")
    
    eval_input = [{
        "formula_jl": formula_jl,
        "q_circuits": [q_circuit],
        "adapt_circuit": [],
        "label": data['instance_filename'],
        "energy_gurobi": data.get("gurobi_energy", 0)
    }]

    in_json = "tmp_reproduce_input.json"
    out_json = "tmp_reproduce_output.json"
    
    with open(in_json, "w") as f:
        json.dump(eval_input, f)

    julia_cmd = [
        "julia",
        "-t", "2",
        "--project=ADAPT.jl", 
        "src/qaoa-gpt/adapt_gpt_eval_energy.jl",
        in_json,
        out_json,
        str(n_nodes),
        pool_type
    ]
    
    result = subprocess.run(julia_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED: Julia process failed\n{result.stderr}")
        return

    with open(out_json, "r") as f:
        res_list = json.load(f)
    
    recon_energy = res_list[0]["adapt_gpt_energies"][0]
    diff = abs(recon_energy - target_energy)
    
    print("\n--- Reproduction Results ---")
    print(f"Target Energy: {target_energy}")
    print(f"Recon Energy:  {recon_energy}")
    print(f"Difference:    {diff:.2e}")
    
    if diff < 1e-12:
        print("\nSUCCESS: Exact energy reproduced!")
    else:
        print("\nFAILURE: Energy mismatch.")

if __name__ == "__main__":
    run_reproduce_test()
