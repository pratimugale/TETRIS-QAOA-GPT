"""
    TetrisConfig

Configuration for Tetris-ADAPT experiments.
"""
Base.@kwdef struct TetrisConfig
    # Physics parameters
    initial_gamma::Float64 = 0.01
    hamiltonian_type::String = "exact" # "exact" or "approximate"
    energy_floor::Float64 = -Inf # Stop if energy <= this value

    # ADAPT-VQE parameters
    gradient_threshold::Float64 = 1e-6
    score_stopper_threshold::Float64 = 1e-6
    parameter_stopper_max::Int = 1000
    layer_stopper_max::Int = 20

    # KaMIS Parameters
    use_kamis::Bool = false
    kamis_seed::Int = 42

    # Slow stopper (convergence check)
    slow_stopper_threshold::Float64 = 1e-6
    slow_stopper_patience::Int = 5
    floor_stopper_threshold::Float64 = 0.1 # Stop if energy is within this threshold of the actual ground state energy

    # Optimizer
    optimizer_tolerance::Float64 = 1e-6
    optimizer_max_iterations::Int = 1000

    # Sampling
    num_shots::Int = 1000

    # Approximation ratio stopping criterion
    # Gurobi benchmarks - used for the ApproxRatioStopper and result plotting
    gurobi_max_satisfied::Float64 = NaN
    gurobi_percent_satisfied::Float64 = NaN
    approx_ratio_stopper_threshold::Float64 = 1
end
