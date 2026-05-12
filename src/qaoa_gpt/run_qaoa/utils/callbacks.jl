"""
    ClauseSatisfactionTracer()

At each adaptation, identify the number of satisfied clauses and save it as an integer.
For now, this function is to be used only for debugging and testing purposes. We will 
be using it to compare the number of satisfied clauses between different methods like 
the approximate and exact Hamiltonians, and the greedy and Kamis methods.
Note that this function is expensive as it is O(2^N) where N is the number of variables.

"""

struct ClauseSatisfactionTracer <: ADAPT.AbstractCallback
    satisfied_clauses::Vector{Int}
    formula_length::Int
end

function ClauseSatisfactionTracer(formula::ADAPT.Hamiltonians.Max3SAT.Types.Formula, n_vars::Int)
    satisfied_clauses = zeros(Int, 2^n_vars)

    # iterate through all possible configurations
    for i in 0:2^n_vars-1
        # convert i to binary bitstring (Vector{Bool}) of length n_vars
        bitstring = Vector{Bool}(digits(i, base=2, pad=n_vars) .== 1)
        satisfied_clauses[i+1] = get_number_of_satisfied_clauses(bitstring, formula)
    end

    return ClauseSatisfactionTracer(satisfied_clauses, length(formula))
end

function (tracer::ClauseSatisfactionTracer)(
    ::ADAPT.Data, ansatz::ADAPT.AbstractAnsatz, trace::ADAPT.Trace,
    ::ADAPT.AdaptProtocol, ::ADAPT.GeneratorList,
    ::ADAPT.Observable, ψ0::ADAPT.QuantumState,
)
    # Final statevector
    ψ = ADAPT.evolve_state(ansatz, ψ0)

    # Calculate the probability of each state P(x) = |ψ(x)|^2
    prob_dist = abs2.(ψ)

    # Expected number of satisfied clauses = sum(P(x) * satisfied_clauses(x))
    expected_satisfied_clauses = sum(prob_dist .* tracer.satisfied_clauses)
    @info "Expected number of satisfied clauses: $(expected_satisfied_clauses)"
    @info "Percentage of satisfied clauses: $(expected_satisfied_clauses / tracer.formula_length)"

    push!(get!(trace, :satisfiedclauses, Any[]), expected_satisfied_clauses)
    return false
end

"""
    ApproxRatioStopper(min_layers, approx_ratio_threshold, gurobi_percent_satisfied_threshold, formula_length)

Stops ADAPT when:
  1. The number of layers added so far exceeds `min_layers`, AND
  2. The approximation ratio exceeds `approx_ratio_threshold`.

Approximation ratio = (expected % satisfied by Tetris) / (% satisfied by Gurobi).

`ClauseSatisfactionTracer` must appear earlier in the callbacks list so that
`:satisfiedclauses` is populated in the trace when this callback runs.
"""
struct ApproxRatioStopper <: ADAPT.AbstractCallback
    approx_ratio_threshold::Float64
    gurobi_percent_satisfied::Float64
    formula_length::Int
end

function (stopper::ApproxRatioStopper)(
    ::ADAPT.Data, ansatz::ADAPT.AbstractAnsatz, trace::ADAPT.Trace,
    ::ADAPT.AdaptProtocol, ::ADAPT.GeneratorList,
)
    # Read the latest expected satisfied clauses (populated by ClauseSatisfactionTracer)
    satisfied_trace = get(trace, :satisfiedclauses, Any[])
    if isempty(satisfied_trace)
        return false
    end
    expected_satisfied = Float64(last(satisfied_trace))

    # Compute approximation ratio
    tetris_percent = expected_satisfied / stopper.formula_length
    approx_ratio = tetris_percent / stopper.gurobi_percent_satisfied

    @info "ApproxRatioStopper: tetris_pct=$(round(tetris_percent, digits=4)), gurobi_pct=$(round(stopper.gurobi_percent_satisfied, digits=4)), ratio=$(round(approx_ratio, digits=4))"

    if approx_ratio >= stopper.approx_ratio_threshold
        @info "ApproxRatioStopper: Stopping — approximation ratio $(round(approx_ratio, digits=4)) ≥ $(stopper.approx_ratio_threshold)."
        trace[:callback_flagged] = "ApproxRatioStopper"
        return true
    end

    return false
end