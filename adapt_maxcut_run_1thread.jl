import Graphs
import SimpleWeightedGraphs
import ADAPT
import PauliOperators: ScaledPauliVector, FixedPhasePauli
import LinearAlgebra: norm
import CSV
import DataFrames
using JuMP, MQLib
using Dates
import JSON
import Random
using ProgressBars
using ArgParse
_DEFAULT_RNG = Random.MersenneTwister(1234);

# Function to parse the command-line arguments
function parse_commandline()
    # Create a new argument parser
    s = ArgParseSettings()

    # Define named arguments
    @add_arg_table s begin
        "--output-dir"
        help = "Output directory"
        arg_type = String
        
        "--graphs-number"
        help = "Number of graphs"
        arg_type = Int
        default = 10
        
        "--trials-per-graph"
        help = "Number of times to run degenerate ADAPT per one graph"
        arg_type = Int
        default = 3
        
        "--n-nodes"
        help = "Number of nodes for generated graphs"
        arg_type = Int
        default = 8
        
        "--weighted"
        help = "Use weighted graphs. If false, all weights are set to 1."
        arg_type = Bool
        default = true
        
        "--use-negative-weights"
        help = "Use negative weights. Even when --weighted=false, weights can still be +1 or -1."
        arg_type = Bool
        default = false
        
        "--run-vqe"
        help = "Run ADAPT VQE"
        arg_type = Bool
        default = false
        
        "--run-qaoa"
        help = "Run ADAPT QAOA"
        arg_type = Bool
        default = true

        "--run-diag-qaoa"
        help = "Run DiagonalQAOA"
        arg_type = Bool
        default = false
        
        "--g0"
        help = "Gamma 0 parameter"
        arg_type = String
        default = "inv"
        
        "--max-layers"
        help = "Cut-off value for number of layers in output circuit"
        arg_type = Int
        default = 30
        
        "--energy-tol-frac"
        help = "Energy tolerance fraction for FloorStopper (if used)"
        arg_type = Float64
        default = 0.001
        
    end

    return parse_args(s)
end

args = parse_commandline()

#####
# Identifiers for the current worker
pid = getpid()
hostname = get(ENV, "HOSTNAME", "unknown")
#println("hostname: ", hostname, "; pid: ", pid)
current_datetime = now()
ts_string = string(Dates.format(current_datetime, "yy-mm-dd__HH_MM"))
#####

# trials_per_graph = 3
# graphs_number = 10
# output_dir = "/lustre/acslab/users/2288/Quantum_stuff/results_darwin/vqe_dataset_parallel_pos_weighted_n10_092024"
# energy_tol_frac = 0.0001
# n_nodes = 10
# max_layers = 30

output_dir = args["output-dir"]
graphs_number = args["graphs-number"]
n_nodes = args["n-nodes"]
trials_per_graph = args["trials-per-graph"]
use_negative_weights = args["use-negative-weights"]
run_vqe = args["run-vqe"]
run_qaoa = args["run-qaoa"]
g0 = args["g0"]
max_layers = args["max-layers"]
energy_tol_frac = args["energy-tol-frac"]
weighted = args["weighted"]
diag_qaoa = args["run-diag-qaoa"]

println("Running ADAPT with parameters (worker: $hostname, pid: $pid):")
for (arg,val) in args
    println("  $arg  =>  $val")
end

#use_negative_weights=false
#println("use_negative_weights: ", use_negative_weights)

function get_weighted_maxcut(g::Graphs.SimpleGraph, rng = _DEFAULT_RNG)
    edge_indices = Graphs.edges(g)
    edge_list = [(Graphs.src(e), Graphs.dst(e), rand(rng, Float64)) for e in edge_indices]
    return edge_list
end

function get_mqlib_energy_from_adj_m(adj_matrix)
    # MQLib block
    mqlib_model = Model(MQLib.Optimizer)
    MQLib.set_heuristic(mqlib_model, "BURER2002")
    Q = adj_matrix
    n_nodes = size(adj_matrix)[1]
    @variable(mqlib_model, x[1:n_nodes], Bin)
    @objective(mqlib_model, Max, x' * Q * x)
    JuMP.optimize!(mqlib_model)
    maxcut_val_mqlib = JuMP.objective_value(mqlib_model)
    
    exact_energy_val = -0.5 * maxcut_val_mqlib

    return exact_energy_val
end

function graph_to_edgelist(g)
    weighted_edge_list = [
        (
            SimpleWeightedGraphs.src(e),
            SimpleWeightedGraphs.dst(e),
            SimpleWeightedGraphs.weight(e)
        ) for e in SimpleWeightedGraphs.edges(g)
    ]

    return weighted_edge_list
end 

function graph_to_adj_m(g)
    adj_matrix = Matrix(SimpleWeightedGraphs.adjacency_matrix(g))
    return adj_matrix
end 

function rand_weigh_graph_generator(n_nodes::Int64, prob::Float64=0.5; weighted::Bool=true, neg_weights::Bool=true)
    
    g = Graphs.erdos_renyi(n_nodes, prob)

    g_weighted = SimpleWeightedGraphs.SimpleWeightedGraph(n_nodes)
    for e in SimpleWeightedGraphs.edges(g)
        s = SimpleWeightedGraphs.src(e)
        d = SimpleWeightedGraphs.dst(e)
        if weighted
            w = round(rand(), digits=2)
            while w == 0.0
                w = round(rand(), digits=2)
            end 
        else
            w = 1
        end 
        if neg_weights
            w_sign = rand([-1,1])
            w = w * w_sign
        end
        SimpleWeightedGraphs.add_edge!(g_weighted, s, d, w)
    end
    return g_weighted
end

function edgelist_to_graph(edgelist, num_vertices=0)

    if num_vertices == 0
        num_vertices = maximum(max(src, dst) for (src, dst, w) in edgelist)
    end
    g = SimpleWeightedGraphs.SimpleWeightedGraph(num_vertices)

    # Add edges with weights to the graph
    for (src, dst, w) in edge_list
        Graphs.add_edge!(g, src, dst, w)
    end
    
    return g
end

if !isdir(""*output_dir*"/hams")
    mkpath(""*output_dir*"/hams")
end
if !isdir(""*output_dir*"/res")
    mkpath(""*output_dir*"/res")
end
if !isdir(""*output_dir*"/graphs")
    mkpath(""*output_dir*"/graphs")
end

start_time = time()

# RUN MANY ADAPT-QAOA TRIALS, CHOOSING RANDOMLY WHEN THE GRADIENTS ARE DEGENERATE
results_df = DataFrames.DataFrame()
hams_df = DataFrames.DataFrame()
graphs_df = DataFrames.DataFrame(
    graph_num = Int[],
    edgelist_json = String[]
)

# main loop

#prob=0.9999

prob_list = 0.3:0.1:0.9  # List of probabilities from 0.3 to 0.9

# BUILD OUT THE POOL
pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_double_pool(n_nodes); pooltype = "qaoa_double_pool"
#pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_mixer(n_nodes); pooltype = "qaoa_mixer"

# ANOTHER POOL OPTION
# pool = ADAPT.Pools.two_local_pool(n_nodes); pooltype = "two_local_pool"

iter = ProgressBar(1:graphs_number)
set_description(iter, "Graphs on: "*hostname*"; pid: "*string(pid)*":")
#for graph_num in 1:graphs_number
for graph_num in iter

    prob = Random.rand(prob_list)
    g = rand_weigh_graph_generator(n_nodes, prob, weighted=weighted, neg_weights=use_negative_weights)

    while Graphs.ne(g) == 0
        println("Generated empty graph! Trying again")
        g = rand_weigh_graph_generator(n_nodes, prob, weighted=weighted, neg_weights=use_negative_weights)
    end
    
    e_list = graph_to_edgelist(g)
    
    edgelist_json = JSON.json(e_list)

    println("Number of edges: ", Graphs.ne(g), "; prob: ", prob)
    push!(graphs_df, (graph_num, edgelist_json))
    
    # BUILD OUT THE PROBLEM HAMILTONIAN
    if diag_qaoa
        H_spv = ADAPT.Hamiltonians.maxcut_hamiltonian(n_nodes, e_list)
        # Wrap in a QAOAObservable view.
        H = ADAPT.ADAPT_QAOA.QAOAObservable(H_spv)
    else
        H = ADAPT.Hamiltonians.maxcut_hamiltonian(n_nodes, e_list)
    end
    
    ##########
    # MQLib block

    exact_energy_val = (
        get_mqlib_energy_from_adj_m(
            graph_to_adj_m(g)
        )
    )
    
    energy_tol = abs(energy_tol_frac * exact_energy_val)

    # println("Generator data type: ", typeof(pool[1]))
    # println("Note: in the current ADAPT-QAOA implementation, the observable and generators must have the same type.")
    
    # SELECT THE PROTOCOLS
    
    #adapt = ADAPT.Degenerate_ADAPT.DEG_ADAPT
    adapt = ADAPT.VANILLA
    vqe = ADAPT.OptimOptimizer(:BFGS; g_tol=1e-4)
    
    # SELECT THE CALLBACKS
    callbacks = [
        ADAPT.Callbacks.Tracer(:energy, :selected_index, :selected_score, :scores, :elapsed_time),
        ADAPT.Callbacks.ParameterTracer(),
        #ADAPT.Callbacks.Printer(:energy, :selected_index, :selected_score),
        ADAPT.Callbacks.ScoreStopper(1e-3),
        ADAPT.Callbacks.ParameterStopper(max_layers * 2),
        ADAPT.Callbacks.FloorStopper(energy_tol, exact_energy_val),
        # ADAPT.Callbacks.SlowStopper(1.0, 3),
        # ADAPT.Callbacks.TimeStopper(soft_time_limit),
    ]
    
    #println("Exact energy (MQLib): $exact_energy_val")
    ##########
    
    for trial_num = 1:trials_per_graph

        ### VQE BLOCK ###
        
        if run_vqe
        
            # INITIALIZE THE REFERENCE STATE
            ψ0 = ones(ComplexF64, 2^n_nodes) / sqrt(2^n_nodes); ψ0 /= norm(ψ0)

            # gamma0=0.5/Graphs.ne(g)

            # println("Using gamma0 = 0.5/Graphs.ne(g) = $gamma0")

            # INITIALIZE THE ANSATZ AND TRACE
            #ansatz = ADAPT.Basics.Ansatz(gamma0, H) 
            ansatz = ADAPT.Basics.Ansatz(1.0, pool) 
            #= the first argument is a hyperparameter and can in principle 
            be set to values other than 0.1 =#
            trace = ADAPT.Trace()

            # RUN THE ALGORITHM
            success = ADAPT.run!(ansatz, trace, adapt, vqe, pool, H, ψ0, callbacks)
            #println(success ? "Success!" : "Failure - optimization didn't converge.")

            # # RESULTS
            # if !success
            #     continue
            # end

            # SAVE THE TRACE
            try
               #throw(error("hello"))
               cur_res_df = DataFrames.DataFrame(
                    :method => "vqe",
                    :graph_num => graph_num,
                    :run => trial_num,
                    :gamma0 => -999.0,
                    :pooltype => pooltype,
                    :generator_index_in_pool => trace[:selected_index][1:end-1], 
                    :β_coeff => -999.0,
                    :γ_coeff => -999.0,
                    :coeff => ansatz.parameters,
                    :energy => trace[:energy][trace[:adaptation][2:end]],
                    :energy_mqlib => exact_energy_val,
                    :took_time => sum(trace[:elapsed_time]),
                )
                append!(results_df, cur_res_df)
            catch err
               @error "ERROR: " exception=(err, catch_backtrace())
            end
        end

        ### QAOA BLOCK ###
        
        if run_qaoa
        
            # INITIALIZE THE REFERENCE STATE
            ψ0 = ones(ComplexF64, 2^n_nodes) / sqrt(2^n_nodes); ψ0 /= norm(ψ0)
            
            if g0 == "inv"
                gamma0=1/Graphs.ne(g)
            else
                gamma0 = parse(Float64, g0)
            end

            println("For QAOA using gamma_0 = $gamma0, trial: $trial_num")

            if diag_qaoa
                ansatz = ADAPT.ADAPT_QAOA.DiagonalQAOAAnsatz(gamma0, pool, H)
            else
                ansatz = ADAPT.ADAPT_QAOA.QAOAAnsatz(gamma0, H)
            end
            #= the first argument is a hyperparameter and can in principle 
            be set to values other than 0.1 =#
            trace = ADAPT.Trace()

            # RUN THE ALGORITHM
            println("RUNNING ADAPT QAOA!!!")
            success = ADAPT.run!(ansatz, trace, adapt, vqe, pool, H, ψ0, callbacks)
            
            #println(success ? "Success!" : "Failure - optimization didn't converge.")

            # # RESULTS
            # if !success
            #     continue
            # end

            # SAVE THE TRACE
            try
               #throw(error("hello"))
               cur_res_df = DataFrames.DataFrame(
                    :method => "qaoa",
                    :graph_num => graph_num,
                    :run => trial_num,
                    :gamma0 => gamma0,
                    :pooltype => pooltype,
                    :generator_index_in_pool => trace[:selected_index][1:end-1], 
                    :β_coeff => ansatz.β_parameters,
                    :γ_coeff => ansatz.γ_parameters,
                    :coeff => -999.0,
                    :energy => trace[:energy][trace[:adaptation][2:end]],
                    :energy_mqlib => exact_energy_val,
                    :took_time => sum(trace[:elapsed_time]),
                )
                append!(results_df, cur_res_df)
            catch err
               @error "ERROR: " exception=(err, catch_backtrace())
            end
        end
    end
    
    H_df = DataFrames.DataFrame(H)
    H_df[!, :graph_num] .= graph_num
    append!(hams_df, H_df)

    # WRITE THE HAMILTONIAN TO A FILE
    ham_file = ""*output_dir*"/hams/n_"*string(n_nodes)*"_worker_"*string(hostname)*"_pid_"*string(pid)*"_ts_"*ts_string*"_Ham.csv"
    CSV.write(ham_file, hams_df)
    
    # WRITE THE ADAPT-QAOA RESULTS TO A FILE
    results_file = ""*output_dir*"/res/n_"*string(n_nodes)*"_worker_"*string(hostname)*"_pid_"*string(pid)*"_ts_"*ts_string*"_adapt_qaoa_res.csv"
    CSV.write(results_file, results_df)
    
    # WRITE GRAPHS TO A FILE
    graphs_file = ""*output_dir*"/graphs/n_"*string(n_nodes)*"_worker_"*string(hostname)*"_pid_"*string(pid)*"_ts_"*ts_string*"_graphs_json.csv"
    CSV.write(graphs_file, graphs_df)
    
end

end_time = time()
elapsed_time = end_time - start_time
println("\nhostname: ", hostname, "; pid: ", pid, "; The script took: ", elapsed_time, " seconds.\n")
