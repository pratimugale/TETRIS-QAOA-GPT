import json
import subprocess
import os

def run_reproduce_test():
    n_nodes = 10
    pool_type = "qaoa_nondiagonal_double_pool"
    target_energy = 0.973298316266657
    
    # Exact formula for random_v10_c51_fd1943f0.cnf
    formula = [
        [1, 2, -3], [1, 2, -9], [1, -2, 4], [1, -3, 5], [1, -3, -6], 
        [1, -4, 6], [1, -4, -6], [1, 5, 6], [1, 5, 10], [1, -5, -9], 
        [1, 7, 9], [1, -9, -10], [-1, 2, 3], [-1, -2, 5], [-1, 3, -4], 
        [-1, -3, 8], [-1, -4, -5], [-1, -4, 9], [-1, -7, 10], [2, 3, 6], 
        [2, 3, -9], [2, -4, -9], [2, -5, 9], [2, -6, -8], [2, 7, -10], 
        [2, -8, 9], [2, -9, -10], [-2, 4, 8], [-2, -4, -9], [-2, -5, -7], 
        [3, 4, 6], [3, -4, -10], [3, -5, 7], [-3, 4, 5], [-3, -6, -7], 
        [4, 5, -7], [4, -6, 10], [4, 9, 10], [-4, 5, 8], [-4, -5, 8], 
        [-4, 6, 7], [-4, -6, -8], [-4, 8, -10], [-5, -6, -9], [-5, -7, -9], 
        [-5, -7, 10], [6, 7, 8], [6, 9, 10], [-6, 7, -10], [7, 8, 10], 
        [7, -8, -9]
    ]
    
    # Selected Mixer Indices (Layer-by-Layer)
    selected_indices = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], # Layer 1
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], # Layer 2
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], # Layer 3
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 20], # Layer 4
        [1, 2, 3, 5, 7, 8, 9, 20, 224]   # Layer 5
    ]
    
    # Optimized Gammas (1 per layer)
    gammas = [
        0.28807388097840186, 0.520303584457635, 
        0.6243235010988054, 0.6839359622424367, 0.8539476306586105
    ]
    
    # Optimized Betas (1 per mixer operator)
    betas = [
        -0.6425202513933348, -0.48529427644602335, -0.6602631694639315, -0.5309102539709725,
        -0.7553875420733934, -0.4526334514822168, -0.5072855919466622, -0.5950105288092726,
        -0.44507105259656865, -0.5881541314540334, -0.4982184253035572, -0.41452025777513263,
        -0.45609304700144265, -0.5078686652805544, -0.6671991260194604, -0.3838939259945674,
        -0.4651587483752416, -0.5165671169431021, -0.38943589044147153, -0.6203496316482581,
        -0.31135325366059347, -0.2842664393939513, -0.33837070805860814, -0.373893686672081,
        -0.6595995671703186, -0.2552790878847646, -0.40386984273988247, -0.37241820297193107,
        -0.28304723138991134, -0.42946660262333014, -0.21371299886913434, -0.20891871317545208,
        -0.32826851485875824, -0.19025474426080927, -0.4214129399801273, -0.19676677658558583,
        -0.369928599236929, -0.28349165670129267, -0.23777016017678204, -0.013690633688405916,
        -0.14817142183454401, -0.08218085343782329, -0.28805953759787806, -0.27054823568143266,
        -0.28055191531490226, -0.13173588442559442, -0.20415361550789865, -0.10766281567615067,
        0.025059461068879584
    ]

    # Convert to Tetris Specification: [new_layer_p, op1, beta1, op2, beta2, ..., gamma]
    q_circuit = []
    beta_ptr = 0
    for i in range(len(gammas)):
        q_circuit.append("new_layer_p")
        for op_idx in selected_indices[i]:
            q_circuit.append(op_idx)
            q_circuit.append(betas[beta_ptr])
            beta_ptr += 1
        q_circuit.append(gammas[i])
        
    print(f"Reproducing exact Tetris-QAOA energy (N={n_nodes})...")
    
    # Wrap for adapt_gpt_eval_energy.jl
    eval_input = [{
        "formula_jl": formula,
        "q_circuits": [q_circuit],
        "adapt_circuit": [],
        "label": "tetris_reproduction_n10",
        "energy_gurobi": 0
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
    
    subprocess.run(julia_cmd, capture_output=True, text=True)

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
