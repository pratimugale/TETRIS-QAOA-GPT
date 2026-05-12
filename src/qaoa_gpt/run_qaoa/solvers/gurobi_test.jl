# This file implements the exact solver for MaxE3SAT using Gurobi.
# It translates the MaxE3SAT problem into a MILP problem and solves it using Gurobi.

include("gurobi_exact_hamiltonian.jl")

clauses = [[-1, -2, -3], [1, -2, 3], [1, 2, -3], [-1, 2, 3], [-1, -2, 3], [-1, 2, -3], [1, 2, 3], [1, -2, -3], [1, 2, 4]]
max_val, state = solve_max_e3sat_exact(4, clauses)

println("Exact Max Satisfied: ", max_val)
println("Ground State Bitstring: ", state)
println("Ground State Energy: ", length(clauses) - max_val)