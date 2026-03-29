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

for graph_idx in 1:length(adapt_gpt_out_list)
    
    adapt_gpt_out_dict = adapt_gpt_out_list[graph_idx]
    formula_data = adapt_gpt_out_dict["formula_jl"];
    adapt_gpt_energies_list = []
    
    for i in 1:(length(adapt_gpt_out_dict["q_circuits"]) + 1)
        if i <= length(adapt_gpt_out_dict["q_circuits"])
            #println(i)
            generated_list = adapt_gpt_out_dict["q_circuits"][i]
        else
            #println("ADAPT")
            generated_list =  adapt_gpt_out_dict["adapt_circuit"]
        end

        E_final = 999 # Default value if sth goes wrong
        try
            E_final = suppress_output(
                eval_ansatz,
                formula_data,
                generated_list,
                n_nodes,
                pool_type,
            )
        catch
        end
        #println(i)
        if i <= length(adapt_gpt_out_dict["q_circuits"])
            append!(adapt_gpt_energies_list, E_final);
        else
            #print("$i ADAPT")
            adapt_gpt_out_dict["ADAPT_energy_round"] = E_final;
        end
    end
    adapt_gpt_out_dict["adapt_gpt_energies"] = adapt_gpt_energies_list;

#     # --- TODO (not needed for now): Generate and Evaluate Null Hypothesis (Random Params) ---
#     # We should maybe test for random parameters generated for valid circuits.
#     null_circuit = copy(adapt_gpt_out_dict["adapt_circuit"])
#     for j in 1:(length(null_circuit)-2)
#         if null_circuit[j] == "new_layer_p"
#             # Replace gamma and beta with random values in range [-10, 10]
#             if j+2 <= length(null_circuit) && null_circuit[j+2] isa Number
#                 null_circuit[j+2] = round(20 * rand() - 10, digits=2)
#             end
#             if j+3 <= length(null_circuit) && null_circuit[j+3] isa Number
#                 null_circuit[j+3] = round(20 * rand() - 10, digits=2)
#             end
#         end
#     end
    
#     e_null = 999
#     try
#         e_null = suppress_output(eval_ansatz, formula_data, null_circuit, n_nodes, pool_type)
#     catch e
#         println("Error in eval_ansatz (Null): $e")
#     end

#     # --- NEW: Calculate Approximation Ratios and Quality ---
#     n_clauses = length(formula_data)
#     e_gurobi = adapt_gpt_out_dict["energy_gurobi"]
#     e_gpt = adapt_gpt_energies_list[1]
#     e_adapt = adapt_gpt_out_dict["ADAPT_energy_round"]

#     # Satisfaction = (Total Clauses - Energy)
#     ar_gpt = (n_clauses - e_gpt) / (n_clauses - e_gurobi)
#     ar_actual = (n_clauses - e_adapt) / (n_clauses - e_gurobi)
#     ar_null = (n_clauses - e_null) / (n_clauses - e_gurobi)
#     quality = ar_gpt / ar_actual

#     adapt_gpt_out_dict["result_quality"] = Dict(
#         "ar_qaoa_gpt" => ar_gpt,
#         "ar_qaoa_actual" => ar_actual,
#         "ar_qaoa_null" => ar_null,
#         "circuit_quality" => quality,
#         "n_clauses" => n_clauses
#     )

#     println("\n--- Performance Metrics (Sample $(graph_idx)) ---")
#     println("Formula Clauses: $n_clauses")
#     println("AR QAOA GPT: $(round(ar_gpt * 100, digits=2))%")
#     println("AR QAOA Actual (ADAPT): $(round(ar_actual * 100, digits=2))%")
#     println("AR QAOA NULL (Random): $(round(ar_null * 100, digits=2))%")
#     println("Circuit Quality (GPT/Actual): $(round(quality * 100, digits=2))%")
#     flush(stdout)
# end

# # --- NEW: Final Aggregate Summary ---
# if length(adapt_gpt_out_list) > 1
#     total_ar_gpt = sum(d["result_quality"]["ar_qaoa_gpt"] for d in adapt_gpt_out_list)
#     total_ar_actual = sum(d["result_quality"]["ar_qaoa_actual"] for d in adapt_gpt_out_list)
#     total_ar_null = sum(d["result_quality"]["ar_qaoa_null"] for d in adapt_gpt_out_list)
#     total_quality = sum(d["result_quality"]["circuit_quality"] for d in adapt_gpt_out_list)
    
#     n_samples = length(adapt_gpt_out_list)
    
#     println("\n" * "="^40)
#     println("FINAL BATCH SUMMARY ($(n_samples) samples)")
#     println("="^40)
#     println("Avg AR QAOA GPT: $(round(total_ar_gpt/n_samples * 100, digits=2))%")
#     println("Avg AR QAOA Actual: $(round(total_ar_actual/n_samples * 100, digits=2))%")
#     println("Avg AR QAOA NULL: $(round(total_ar_null/n_samples * 100, digits=2))%")
#     println("Avg Circuit Quality: $(round(total_quality/n_samples * 100, digits=2))%")
#     println("="^40)
end

## Saving

adapt_gpt_out_list_json = JSON.json(adapt_gpt_out_list);

open(output_fpath,"w") do f 
    write(f, adapt_gpt_out_list_json) 
end