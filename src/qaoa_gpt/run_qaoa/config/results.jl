import JSON

"""
    TetrisResult

Container for the results of a Tetris-ADAPT execution.
"""
Base.@kwdef mutable struct TetrisResult
    instance_id::Int
    method::String
    instance_filename::String = ""
    success::Bool
    callback_flagged::String = ""

    # Timing (seconds)
    total_runtime::Float64 = 0.0
    hamiltonian_construction_time::Float64 = 0.0
    ansatz_creation_time::Float64 = 0.0
    adapt_runtime::Float64 = 0.0
    sampling_time::Float64 = 0.0

    final_energy::Float64
    hamiltonian_terms::Int

    # Instance Stats (populated by experiment)
    num_clauses::Int = 0
    percent_satisfied_clauses::Float64 = 0.0

    # Traces
    num_adapt_layers::Int = 0
    num_iterations::Int = 0
    selected_indices::Vector{Any} = Any[] # Can be Int or Vector{Int} depending on pool
    selected_scores::Any = []
    selected_operator_strings::Vector{String} = String[]

    # Ansatz Structure
    gamma_values::Vector{Float64} = Float64[]
    beta_values::Vector{Float64} = Float64[]

    # Sampling results
    sampled_expected_satisfaction::Float64
    sampled_best_satisfaction::Int
    sampled_best_solution::Vector{Bool}

    # Tracking History
    adaptation_energies::Vector{Float64} = Float64[]
    parameter_trace::Any = []

    # Analysis
    first_layer_gradient_sum::Float64 = 0.0
    max_pool_gradients::Vector{Float64} = Float64[]
    clause_satisfaction_percent_trace::Vector{Float64} = Float64[]

    # Gurobi benchmarks
    initial_gamma::Float64 = NaN
    gurobi_energy::Float64 = NaN
    approximation_ratio::Float64 = NaN
end

"""
    BruteForceResult

Container for the results of a Brute Force execution.
"""
Base.@kwdef struct BruteForceResult
    instance_id::Int
    method::String = "bruteforce"
    best_solution::Vector{Bool}
    best_satisfaction_count::Int
    percent_satisfied_clauses::Float64
    approx_hamiltonian_energy::Float64
    execution_time::Float64
end

"""
    BenchmarkResult

Container for the combined results of Brute Force and Greedy Tetris execution.
"""
Base.@kwdef struct BenchmarkResult
    instance_id::Int
    n_vars::Int
    bruteforce_result::BruteForceResult
    tetris_result::TetrisResult
end
