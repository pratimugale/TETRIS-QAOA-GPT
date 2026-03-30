import Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

ENV["OPENBLAS_NUM_THREADS"] = "1"
import ADAPT
import CSV
import DataFrames
import DataFrames: groupby
import Serialization
import LinearAlgebra: norm
import JSON

include("eval_ansatz.jl")

function run_evaluation()
    if length(ARGS) < 4
        println("Usage: julia adapt_gpt_eval_energy.jl <input_fpath> <output_fpath> <n_nodes> <pool_type>")
        exit(1)
    end

    input_fpath = ARGS[1]
    output_fpath = ARGS[2]
    n_nodes = parse(Int, ARGS[3])
    pool_type = ARGS[4]

    adapt_gpt_input_json = JSON.parsefile(input_fpath)
    # Support both raw list and metadata-wrapped formats
    if adapt_gpt_input_json isa Dict && haskey(adapt_gpt_input_json, "data")
        adapt_gpt_out_list = adapt_gpt_input_json["data"]
    else
        adapt_gpt_out_list = adapt_gpt_input_json
    end

    n_samples = length(adapt_gpt_out_list)

    # Dictionary to store stratified results
    # Key: type (e.g., "balanced", "random")
    # Value: Dict( total_ar_gpt, total_ar_actual, total_quality, n_valid, n_total )
    stratified_metrics = Dict{String, Dict{String, Any}}()

    for graph_idx in 1:n_samples
        
        adapt_gpt_out_dict = adapt_gpt_out_list[graph_idx]
        formula_data = adapt_gpt_out_dict["formula_jl"];
        formula_type = get(adapt_gpt_out_dict, "type", "unknown")
        
        # Initialize metrics for this type if not already seen
        if !haskey(stratified_metrics, formula_type)
            stratified_metrics[formula_type] = Dict(
                "total_ar_gpt" => 0.0,
                "total_ar_actual" => 0.0,
                "total_quality" => 0.0,
                "n_valid" => 0,
                "n_total" => 0
            )
        end
        stratified_metrics[formula_type]["n_total"] += 1

        adapt_gpt_energies_list = []
        
        # 1. Evaluate GPT generated circuits (and ground truth)
        for i in 1:(length(adapt_gpt_out_dict["q_circuits"]) + 1)
            is_gpt = i <= length(adapt_gpt_out_dict["q_circuits"])
            
            if is_gpt
                generated_list = adapt_gpt_out_dict["q_circuits"][i]
            else
                generated_list = adapt_gpt_out_dict["adapt_circuit"]
            end

            E_final = 999 
            try
                E_final = suppress_output(eval_ansatz, formula_data, generated_list, n_nodes, pool_type)
            catch e
                # Circuit fails parse or simulation
            end
            
            if is_gpt
                append!(adapt_gpt_energies_list, E_final);
                if E_final < 900
                    stratified_metrics[formula_type]["n_valid"] += 1
                end
            else
                adapt_gpt_out_dict["ADAPT_energy_round"] = E_final;
            end
        end
        adapt_gpt_out_dict["adapt_gpt_energies"] = adapt_gpt_energies_list;

        # 2. Performance Metrics for the sample
        n_clauses = length(formula_data)
        e_gurobi = adapt_gpt_out_dict["energy_gurobi"]
        e_gpt = adapt_gpt_energies_list[1]
        e_adapt = adapt_gpt_out_dict["ADAPT_energy_round"]

        is_circuit_valid = e_gpt < 900
        if is_circuit_valid
            ar_gpt = (n_clauses - e_gpt) / (n_clauses - e_gurobi)
        else
            ar_gpt = 0.0
        end
        
        ar_actual = (n_clauses - e_adapt) / (n_clauses - e_gurobi)
        quality = is_circuit_valid ? (ar_gpt / ar_actual) : 0.0

        stratified_metrics[formula_type]["total_ar_gpt"] += ar_gpt
        stratified_metrics[formula_type]["total_ar_actual"] += ar_actual
        stratified_metrics[formula_type]["total_quality"] += quality

        adapt_gpt_out_dict["result_quality"] = Dict(
            "ar_qaoa_gpt" => ar_gpt,
            "ar_qaoa_actual" => ar_actual,
            "circuit_quality" => quality,
            "n_clauses" => n_clauses,
            "is_valid" => is_circuit_valid
        )

        println("\n--- Performance Metrics (Sample $(graph_idx), Type: $(formula_type)) ---")
        println("Formula Clauses: $n_clauses")
        if is_circuit_valid
            println("AR QAOA GPT: $(round(ar_gpt * 100, digits=2))%")
            println("AR QAOA Actual (ADAPT): $(round(ar_actual * 100, digits=2))%")
            println("Circuit Quality (GPT/Actual): $(round(quality * 100, digits=2))%")
        else
            println("AR QAOA GPT: [INVALID CIRCUIT - FAILS PARSE]")
        end
        flush(stdout)
    end

    # 3. Final Aggregate Summary
    println("\n" * "="^60)
    println("FINAL STRATIFIED SUMMARY")
    println("="^60)
    
    # Sort types if possible
    types = sort(collect(keys(stratified_metrics)))
    
    total_samples = 0
    global_total_ar_gpt = 0.0
    global_total_ar_actual = 0.0
    global_total_quality = 0.0
    global_n_valid = 0

    for t in types
        m = stratified_metrics[t]
        n_t = m["n_total"]
        n_v = m["n_valid"]
        svr = (n_v / n_t) * 100
        avg_ar_gpt_val = n_v > 0 ? (m["total_ar_gpt"] / n_v) : 0.0
        avg_ar_gpt_all = m["total_ar_gpt"] / n_t
        avg_ar_actual = m["total_ar_actual"] / n_t
        avg_quality = m["total_quality"] / n_t
        
        println("TYPE: $(uppercase(t)) ($(n_t) samples)")
        println("-"^30)
        println("Avg AR QAOA GPT (Valid Only): $(round(avg_ar_gpt_val * 100, digits=2))%")
        println("Avg AR QAOA GPT (Overall):    $(round(avg_ar_gpt_all * 100, digits=2))%")
        println("Avg AR QAOA Actual:           $(round(avg_ar_actual * 100, digits=2))%")
        println("Avg Circuit Quality:          $(round(avg_quality * 100, digits=2))%")
        println("Structural Validity (SVR):     $(round(svr, digits=2))%")
        println("-"^30)
        
        # Aggregate to global
        total_samples += n_t
        global_total_ar_gpt += m["total_ar_gpt"]
        global_total_ar_actual += m["total_ar_actual"]
        global_total_quality += m["total_quality"]
        global_n_valid += n_v
    end

    if total_samples > 1
        println("\nGLOBAL SUMMARY ($(total_samples) samples)")
        println("-"^30)
        global_avg_ar_gpt_val = global_n_valid > 0 ? (global_total_ar_gpt / global_n_valid) : 0.0
        global_avg_ar_gpt_all = global_total_ar_gpt / total_samples
        
        println("Avg AR QAOA GPT (Valid Only): $(round(global_avg_ar_gpt_val * 100, digits=2))%")
        println("Avg AR QAOA GPT (Overall):    $(round(global_avg_ar_gpt_all * 100, digits=2))%")
        println("Avg AR QAOA Actual:           $(round(global_total_ar_actual/total_samples * 100, digits=2))%")
        println("Structural Validity (SVR):     $(round((global_n_valid/total_samples)*100, digits=2))%")
    end
    println("="^60)

    ## Saving
    adapt_gpt_out_list_json = JSON.json(adapt_gpt_out_list);
    open(output_fpath,"w") do f 
        write(f, adapt_gpt_out_list_json) 
    end
end

run_evaluation()