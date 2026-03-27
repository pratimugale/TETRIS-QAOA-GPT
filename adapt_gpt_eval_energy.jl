import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

ENV["OPENBLAS_NUM_THREADS"] = "1"
import ADAPT
import CSV
import DataFrames
import DataFrames: groupby
import Serialization
import LinearAlgebra: norm
import JSON
using ProgressBars
using Base.Threads

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

iter = ProgressBar(1:length(adapt_gpt_out_list))

@threads for graph_idx in iter
    
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
        catch e
            println("Error in eval_ansatz: $e")
            rethrow(e)
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
end

## Saving

adapt_gpt_out_list_json = JSON.json(adapt_gpt_out_list);

open(output_fpath,"w") do f 
    write(f, adapt_gpt_out_list_json) 
end