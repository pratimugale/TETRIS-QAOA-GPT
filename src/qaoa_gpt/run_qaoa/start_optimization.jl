import Pkg
Pkg.activate(".")

import JSON
using Printf
using LinearAlgebra

include(joinpath(@__DIR__, "ADAPT_QAOA_EXPERIMENTS.jl"))
import .ADAPT_QAOA_EXPERIMENTS: TetrisConfig, TetrisResult, parse_cnf_file, run_tetris

# note that method_name = "kamis" corresponds to mosaicadapt_qaoa and method_name = "greedy" corresponds to tetris_qaoa
# This is because we use the KaMIS solver underneath to maximize the sum of gradients of operators in the pool
function run_multi_gamma(input_file::String, output_dir::String, method_name::String)
    # The following is required to prevent multithreading issues on HPC.
    # We want exactly 1 Julia thread to run the optimization so that we can parallelize 
    #  the optimization at a process level.
    ENV["JULIA_NUM_THREADS"] = "1"
    LinearAlgebra.BLAS.set_num_threads(1)
    mkpath(output_dir)

    layer_limit = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 20
    pool_type = length(ARGS) >= 6 ? ARGS[6] : "qaoa_nondiagonal_double_pool"
    
    # Define the gammas we want to sweep for each instance
    gammas = [0.01, 0.1, 0.5]

    for gamma in gammas
        println(">>> Running $method_name with initial_gamma=$gamma, pool=$pool_type, and layer_limit=$layer_limit on $input_file")
        
        instance = parse_cnf_file(input_file)
        instance["instance_id"] = 0 
        
        # 1. Run Gurobi to find the actual ground truth for this instance
        println(">>> Running Gurobi exact solver for Ground Truth...")
        clauses_raw = ADAPT_QAOA_EXPERIMENTS.get_formula_as_list(instance["formula"])
        max_sat, _ = ADAPT_QAOA_EXPERIMENTS.solve_max_e3sat_exact(instance["variables"], clauses_raw)
        
        num_clauses = length(instance["formula"])
        gt_energy = Float64(num_clauses - max_sat)
        g_pct = max_sat / num_clauses
        println(">>> Gurobi Result: Max Satisfied=$max_sat / $num_clauses (pct=$g_pct, energy_min=$gt_energy)")

        config = TetrisConfig(
            initial_gamma=gamma,
            optimizer_tolerance=1e-6,
            gradient_threshold=1e-6,
            score_stopper_threshold=1e-6,
            slow_stopper_threshold=1e-6,
            layer_stopper_max=layer_limit,
            num_shots=1000,
            hamiltonian_type="exact",
            approx_ratio_stopper_threshold=0.999,
            gurobi_max_satisfied=Float64(max_sat),
            gurobi_percent_satisfied=g_pct,
            energy_floor=gt_energy,
            parameter_stopper_max=1000
        )

        use_kamis = method_name == "kamis"
        res = run_tetris(config, instance, pool_type=pool_type, use_kamis=use_kamis, method_name="tetris_adapt_qaoa_$method_name", instance_filename=basename(input_file))

        # All instances are perfectly satisfiable, so the exact ratio should reach 1.0 (or >0.999).
        label = replace(basename(input_file), ".cnf" => "")
        filename = "qaoa_tetris_adapt_qaoa_$(method_name)_gamma$(gamma)_$(label).json"
        
        # Strip the massive dense parameter traces 
        res.parameter_trace = []
        
        open(joinpath(output_dir, filename), "w") do f
            JSON.print(f, res, 2)
        end
        println("Saved result to $(joinpath(output_dir, filename))")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 4
        println("Usage: julia src/qaoa/start_optimization.jl <input_cnf> <output_dir> <gamma_ignored> <method_name> [layer_limit]")
        exit(1)
    end
    
    input_file = ARGS[1]
    output_dir = ARGS[2]
    # gamma = parse(Float64, ARGS[3]) # Ignored now that we loop inside
    method_name = ARGS[4] # either "greedy" or "kamis"
    # layer_limit is handled inside run_multi_gamma via ARGS

    run_mosaic_grid_search(input_file, output_dir, method_name)
end
