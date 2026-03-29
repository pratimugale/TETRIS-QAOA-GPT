import json
import sys
import os
import subprocess

# Add current directory and submodule to path
sys.path.append(os.getcwd())

# Representative test cases extracted from actual optimized 12-variable Max-3-SAT optimized circuits
HARDCODED_TEST_CASES = [
    {
        "filename": "balanced_v12_c41_2603cc48.cnf",
        "formula_jl": [[1, -2, 4], [1, -5, -12], [1, 7, -12], [1, 9, 11], [-1, -3, -6], [-1, -3, -11], [-1, -7, 12], [-1, 8, 10], [-1, -11, 12], [2, -3, -7], [2, -3, 8], [2, -3, 12], [2, 5, 10], [2, -6, -11], [2, -7, 10], [-2, 3, 4], [-2, 3, -5], [-2, -8, 9], [-2, -8, -10], [3, 4, 6], [3, -10, 11], [-3, 8, -9], [4, 6, 7], [4, 7, -8], [-4, 5, 8], [-4, 5, -9], [-4, 5, 10], [-4, -7, -11], [-4, 8, 12], [-4, -9, 12], [5, -6, -9], [5, 8, -11], [-5, 7, 9], [-5, 9, 11], [6, 7, 9], [6, -8, -12], [6, -10, -12], [-6, -7, -9], [-6, -7, 10], [-6, -9, 10], [-8, -10, 11]],
        "gammas": [-0.047070276601532526, 0.30859955026541047, 0.616679294195089, 0.7279579535437108, 0.9441226624961653],
        "betas": [-10.719838724461711, -0.36801774239534385, -0.2756644282356073, -0.20815655778655318, -0.12862628186053218],
        "op_list": [1, 1, 1, 1, 1],
        "expected_energy": 1.5277786271070164,
        "energy_gurobi": 1.0,
        "ADAPT_energy_round": 1.5277786271070164
    },
    {
        "filename": "random_v12_c55_6c8cbc98.cnf",
        "formula_jl": [[1, 2, -8], [1, 3, -5], [1, 3, 7], [1, 3, 11], [1, -4, -12], [-1, 2, -4], [-1, 2, 5], [-1, 2, 8], [-1, -3, 11], [-1, 4, 6], [-1, -4, -8], [-1, 5, -11], [-1, -6, -12], [-1, 8, -9], [-1, -8, -11], [2, -4, -8], [2, -5, 7], [2, -7, 12], [2, -10, -11], [-2, 7, -8], [-2, -7, -9], [-2, 10, 12], [3, 4, -6], [3, -4, -7], [3, 5, 12], [3, -5, 11], [3, 6, -9], [3, -8, 9], [3, -9, -10], [-3, 4, -9], [-3, 4, -12], [-3, -4, -12], [-3, -6, -12], [-3, 7, -12], [4, -5, 8], [4, 7, -8], [4, -8, -9], [-4, 8, 10], [-4, -8, 11], [5, 6, 7], [5, -6, -8], [5, -8, -11], [5, 9, -12], [-5, 6, 9], [6, 8, -9], [6, 9, 10], [6, 11, 12], [-6, -7, 10], [-6, -11, 12], [7, 8, -9], [7, -8, -12], [-7, -8, 11], [-7, -10, 11], [-7, -10, 12], [-8, 9, -10]],
        "gammas": [0.2383902702297476, 0.44171746797997485, 0.5692358039319985, 0.6928504324516042, 0.8011724790608945],
        "betas": [-0.5920322245246821, -0.4894778776974597, -0.3837545075586038, -0.29538003122665146, -0.1745281986315709],
        "op_list": [1, 1, 1, 1, 1],
        "expected_energy": 1.1873657868230623,
        "energy_gurobi": 0.0,
        "ADAPT_energy_round": 1.1873657868230623
    },
    {
        # Test case - this is a copy of the above circuit. It contains a "new_layer_p" placeholder for the first layer's gamma.
        # this is to test if the script correct reports circuit errors
        # The energy returned in this error case will be 999 which we assert for
        "filename": "random_v12_c55_6c8cbc98.cnf",
        "formula_jl": [[1, 2, -8], [1, 3, -5], [1, 3, 7], [1, 3, 11], [1, -4, -12], [-1, 2, -4], [-1, 2, 5], [-1, 2, 8], [-1, -3, 11], [-1, 4, 6], [-1, -4, -8], [-1, 5, -11], [-1, -6, -12], [-1, 8, -9], [-1, -8, -11], [2, -4, -8], [2, -5, 7], [2, -7, 12], [2, -10, -11], [-2, 7, -8], [-2, -7, -9], [-2, 10, 12], [3, 4, -6], [3, -4, -7], [3, 5, 12], [3, -5, 11], [3, 6, -9], [3, -8, 9], [3, -9, -10], [-3, 4, -9], [-3, 4, -12], [-3, -4, -12], [-3, -6, -12], [-3, 7, -12], [4, -5, 8], [4, 7, -8], [4, -8, -9], [-4, 8, 10], [-4, -8, 11], [5, 6, 7], [5, -6, -8], [5, -8, -11], [5, 9, -12], [-5, 6, 9], [6, 8, -9], [6, 9, 10], [6, 11, 12], [-6, -7, 10], [-6, -11, 12], [7, 8, -9], [7, -8, -12], [-7, -8, 11], [-7, -10, 11], [-7, -10, 12], [-8, 9, -10]],
        "gammas": ["new_layer_p", 0.44171746797997485, 0.5692358039319985, 0.6928504324516042, 0.8011724790608945],
        "betas": [-0.5920322245246821, -0.4894778776974597, -0.3837545075586038, -0.29538003122665146, -0.1745281986315709],
        "op_list": [1, 1, 1, 1, 1],
        "expected_energy": 999,
        "energy_gurobi": 999,
        "ADAPT_energy_round": 999
    },
    {
        # another negative test case - here the operator index is out of the range of the pool indices.
        "filename": "random_v12_c55_6c8cbc98.cnf",
        "formula_jl": [[1, 2, -8], [1, 3, -5], [1, 3, 7], [1, 3, 11], [1, -4, -12], [-1, 2, -4], [-1, 2, 5], [-1, 2, 8], [-1, -3, 11], [-1, 4, 6], [-1, -4, -8], [-1, 5, -11], [-1, -6, -12], [-1, 8, -9], [-1, -8, -11], [2, -4, -8], [2, -5, 7], [2, -7, 12], [2, -10, -11], [-2, 7, -8], [-2, -7, -9], [-2, 10, 12], [3, 4, -6], [3, -4, -7], [3, 5, 12], [3, -5, 11], [3, 6, -9], [3, -8, 9], [3, -9, -10], [-3, 4, -9], [-3, 4, -12], [-3, -4, -12], [-3, -6, -12], [-3, 7, -12], [4, -5, 8], [4, 7, -8], [4, -8, -9], [-4, 8, 10], [-4, -8, 11], [5, 6, 7], [5, -6, -8], [5, -8, -11], [5, 9, -12], [-5, 6, 9], [6, 8, -9], [6, 9, 10], [6, 11, 12], [-6, -7, 10], [-6, -11, 12], [7, 8, -9], [7, -8, -12], [-7, -8, 11], [-7, -10, 11], [-7, -10, 12], [-8, 9, -10]],
        "gammas": [0.2383902702297476, 0.44171746797997485, 0.5692358039319985, 0.6928504324516042, 0.8011724790608945],
        "betas": [-0.5920322245246821, -0.4894778776974597, -0.3837545075586038, -0.29538003122665146, -0.1745281986315709],
        "op_list": [2, 1, 1, 1, 1],
        "expected_energy": 999,
        "energy_gurobi": 999,
        "ADAPT_energy_round": 999
    },
]

def run_tests():
    n_nodes = 12
    pool_type = "qaoa_mixer"
    
    print(f"Running Julia evaluation for {len(HARDCODED_TEST_CASES)} hardcoded cases...")
    
    # Prepare data for Julia
    eval_input = []
    for case in HARDCODED_TEST_CASES:
        n_layers = len(case['op_list'])
        q_circuit = []
        for i in range(n_layers):
            q_circuit.append("new_layer_p") 
            q_circuit.append(case['op_list'][i])
            q_circuit.append(case['betas'][i])
            q_circuit.append(case['gammas'][i])
            
        eval_input.append({
            "formula_jl": case['formula_jl'],
            "q_circuits": [q_circuit],
            "adapt_circuit": [],
            "label": case['filename'],
            "energy_gurobi": case['energy_gurobi'],
            "ADAPT_energy_round": case['ADAPT_energy_round']
        })

    in_json = "tmp_input.json"
    out_json = "tmp_output.json"
    
    with open(in_json, "w") as f:
        json.dump(eval_input, f)

    # julia -t n adapt_gpt_eval_energy.jl in out n_nodes pool
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
    
    print(f"Command: {' '.join(julia_cmd)}")
    result = subprocess.run(julia_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Julia process failed with return code {result.returncode}")
        print("\n--- Julia Stderr ---")
        print(result.stderr)
        return

    # Load result
    with open(out_json, "r") as f:
        res_list = json.load(f)
    
    print("\n--- Results ---")
    all_passed = True
    for i, case in enumerate(HARDCODED_TEST_CASES):
        reconstructed_energy = res_list[i]["adapt_gpt_energies"][0]
        orig_energy = case["expected_energy"]
        diff = abs(reconstructed_energy - orig_energy)
        
        status = "PASSED" if diff < 1e-6 else "FAILED"
        print(f"[{status}] Case: {case['filename']}")
        print(f"      Target Energy: {orig_energy}")
        print(f"      Recon Energy:  {reconstructed_energy}")
        print(f"      Diff:          {diff:.2e}")
        
        if status == "FAILED":
            all_passed = False
            
    if all_passed:
        print("\nSUCCESS: All hardcoded reconstructions match perfectly!")
    else:
        print("\nERROR: Some reconstructions failed.")

if __name__ == "__main__":
    run_tests()
