module ADAPT_QAOA_EXPERIMENTS

import ADAPT

# Export core types
export TetrisConfig, TetrisResult, BruteForceResult, BenchmarkResult
export run_tetris, run_greedy_tetris, run_bruteforce, run_tetris_vqe, run_vanilla_qaoa, run_vanilla_vqe, parse_cnf_file
export ClauseSatisfactionTracer, ApproxRatioStopper, get_formula_as_list, solve_max_e3sat_exact

include("config/config.jl")
include("config/results.jl")

# Utilities
include("utils/max3sat.jl")
include("utils/pauli_utils.jl")
include("utils/callbacks.jl")

# Exact Solvers
include("solvers/gurobi.jl")

# Include Runners
include("mosaicadapt_qaoa.jl")
end # module ADAPT_QAOA_EXPERIMENTS
