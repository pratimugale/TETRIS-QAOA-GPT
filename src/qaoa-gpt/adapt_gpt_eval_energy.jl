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
    stratified_metrics = Dict{String, Dict{String, Any}}()

    for graph_idx in 1:n_samples
        adapt_gpt_out_dict = adapt_gpt_out_list[graph_idx]
        formula_data = adapt_gpt_out_dict["formula_jl"];
        formula_type = get(adapt_gpt_out_dict, "type", "unknown")
        
        if !haskey(stratified_metrics, formula_type)
            stratified_metrics[formula_type] = Dict(
                "total_best_ar_gpt" => 0.0,
                "total_avg_ar_gpt" => 0.0,
                "total_ar_actual" => 0.0,
                "n_formulas_with_valid" => 0,
                "n_total_formulas" => 0,
                "n_valid_circuits" => 0,
                "n_total_circuits" => 0
            )
        end
        stratified_metrics[formula_type]["n_total_formulas"] += 1

        adapt_gpt_energies_list = []
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
            else
                adapt_gpt_out_dict["ADAPT_energy_round"] = E_final;
            end
        end
        adapt_gpt_out_dict["adapt_gpt_energies"] = adapt_gpt_energies_list;

        n_clauses = length(formula_data)
        e_gurobi = 0
        if haskey(adapt_gpt_out_dict, "energy_gurobi")
            e_gurobi = adapt_gpt_out_dict["energy_gurobi"]
        elseif haskey(adapt_gpt_out_dict, "ground_truth_energy")
            e_gurobi = adapt_gpt_out_dict["ground_truth_energy"]
        elseif haskey(adapt_gpt_out_dict, "energy_mqlib")
            e_gurobi = adapt_gpt_out_dict["energy_mqlib"]
        end
        
        e_adapt = adapt_gpt_out_dict["ADAPT_energy_round"]
        ar_actual = (n_clauses - e_adapt) / (n_clauses - e_gurobi)

        valid_energies = filter(e -> e < 900, adapt_gpt_energies_list)
        n_total_circuits = length(adapt_gpt_energies_list)
        n_valid_circuits = length(valid_energies)

        stratified_metrics[formula_type]["n_total_circuits"] += n_total_circuits
        stratified_metrics[formula_type]["n_valid_circuits"] += n_valid_circuits

        if n_valid_circuits > 0
            best_e_gpt = minimum(valid_energies)
            avg_e_gpt = sum(valid_energies) / n_valid_circuits
            
            ar_gpt_best = (n_clauses - best_e_gpt) / (n_clauses - e_gurobi)
            ar_gpt_avg = (n_clauses - avg_e_gpt) / (n_clauses - e_gurobi)
            
            stratified_metrics[formula_type]["n_formulas_with_valid"] += 1
            stratified_metrics[formula_type]["total_best_ar_gpt"] += ar_gpt_best
            stratified_metrics[formula_type]["total_avg_ar_gpt"] += ar_gpt_avg
        else
            ar_gpt_best = 0.0
            ar_gpt_avg = 0.0
        end
        
        stratified_metrics[formula_type]["total_ar_actual"] += ar_actual

        adapt_gpt_out_dict["result_quality"] = Dict(
            "best_ar_qaoa_gpt" => ar_gpt_best,
            "avg_ar_qaoa_gpt" => ar_gpt_avg,
            "ar_qaoa_actual" => ar_actual,
            "n_clauses" => n_clauses,
            "sample_svr" => n_total_circuits > 0 ? (n_valid_circuits / n_total_circuits) : 0.0
        )

        println("\n--- Performance Metrics (Formula $(graph_idx), Type: $(formula_type)) ---")
        println("Formula Clauses: $n_clauses | Samples generated: $n_total_circuits")
        println("Valid Circuits: $n_valid_circuits / $n_total_circuits")
        if n_valid_circuits > 0
            println("Best AR QAOA GPT: $(round(ar_gpt_best * 100, digits=2))%")
            println("Avg  AR QAOA GPT: $(round(ar_gpt_avg * 100, digits=2))%")
            println("AR QAOA Actual:   $(round(ar_actual * 100, digits=2))%")
        else
            println("AR QAOA GPT: [ALL CIRCUITS INVALID]")
        end
        flush(stdout)
    end

    # 3. Final Aggregate Summary
    println("\n" * "="^60)
    println("FINAL STRATIFIED SUMMARY")
    println("="^60)
    
    types = sort(collect(keys(stratified_metrics)))
    
    global_n_f = 0
    global_n_f_v = 0
    global_n_c = 0
    global_n_c_v = 0
    global_total_best_ar = 0.0
    global_total_avg_ar = 0.0
    global_total_ar_actual = 0.0

    for t in types
        m = stratified_metrics[t]
        n_f = m["n_total_formulas"]
        n_f_v = m["n_formulas_with_valid"]
        n_c = m["n_total_circuits"]
        n_c_v = m["n_valid_circuits"]
        
        avg_svr = n_c > 0 ? (n_c_v / n_c) * 100 : 0.0
        best_svr = n_f > 0 ? (n_f_v / n_f) * 100 : 0.0
        
        avg_er = 100.0 - avg_svr
        best_er = 100.0 - best_svr
        
        best_ar_val = n_f_v > 0 ? (m["total_best_ar_gpt"] / n_f_v) : 0.0
        best_ar_all = n_f > 0 ? (m["total_best_ar_gpt"] / n_f) : 0.0
        
        avg_ar_val = n_f_v > 0 ? (m["total_avg_ar_gpt"] / n_f_v) : 0.0
        avg_ar_all = n_f > 0 ? (m["total_avg_ar_gpt"] / n_f) : 0.0
        
        avg_ar_act = n_f > 0 ? (m["total_ar_actual"] / n_f) : 0.0
        
        println("TYPE: $(uppercase(t)) ($(n_f) formulas)")
        println("-"^30)
        println("Best ER (Formula failure): $(round(best_er, digits=2))%")
        println("Avg  ER (Circuit failure): $(round(avg_er, digits=2))%")
        println("Best AR QAOA GPT (Valid):  $(round(best_ar_val * 100, digits=2))%")
        println("Best AR QAOA GPT (All):    $(round(best_ar_all * 100, digits=2))%")
        println("Avg  AR QAOA GPT (Valid):  $(round(avg_ar_val * 100, digits=2))%")
        println("Avg  AR QAOA GPT (All):    $(round(avg_ar_all * 100, digits=2))%")
        println("AR QAOA Actual:            $(round(avg_ar_act * 100, digits=2))%")
        println("-"^30)
        
        global_n_f += n_f
        global_n_f_v += n_f_v
        global_n_c += n_c
        global_n_c_v += n_c_v
        global_total_best_ar += m["total_best_ar_gpt"]
        global_total_avg_ar += m["total_avg_ar_gpt"]
        global_total_ar_actual += m["total_ar_actual"]
    end

    if global_n_f > 1
        println("\nGLOBAL SUMMARY ($(global_n_f) formulas, $(global_n_c) circuits)")
        println("-"^30)
        global_avg_svr = (global_n_c_v / global_n_c) * 100
        global_best_svr = (global_n_f_v / global_n_f) * 100
        
        global_avg_er = 100.0 - global_avg_svr
        global_best_er = 100.0 - global_best_svr
        
        g_best_ar_val = global_n_f_v > 0 ? (global_total_best_ar / global_n_f_v) : 0.0
        g_best_ar_all = global_total_best_ar / global_n_f
        
        g_avg_ar_val = global_n_f_v > 0 ? (global_total_avg_ar / global_n_f_v) : 0.0
        g_avg_ar_all = global_total_avg_ar / global_n_f
        
        println("Best ER (Formula failure): $(round(global_best_er, digits=2))%")
        println("Avg  ER (Circuit failure): $(round(global_avg_er, digits=2))%")
        println("Best AR QAOA GPT (Valid):  $(round(g_best_ar_val * 100, digits=2))%")
        println("Best AR QAOA GPT (All):    $(round(g_best_ar_all * 100, digits=2))%")
        println("Avg  AR QAOA GPT (Valid):  $(round(g_avg_ar_val * 100, digits=2))%")
        println("Avg  AR QAOA GPT (All):    $(round(g_avg_ar_all * 100, digits=2))%")
        println("AR QAOA Actual:            $(round((global_total_ar_actual/global_n_f) * 100, digits=2))%")
    end
    println("="^60)

    ## Saving
    adapt_gpt_out_list_json = JSON.json(adapt_gpt_out_list);
    open(output_fpath,"w") do f 
        write(f, adapt_gpt_out_list_json) 
    end
end

run_evaluation()