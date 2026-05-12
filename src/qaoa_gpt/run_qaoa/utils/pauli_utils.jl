
import PauliOperators: ScaledPauliVector, ScaledPauli

"""
    pauli_op_to_string(op, n_vars) -> String

Convert a pool operator (Vector{ScaledPauli}) to a human-readable Pauli string.
"""
function pauli_op_to_string(op, n_vars::Int)::String
    term_strings = String[]
    for sp in op
        pauli = sp.pauli
        qubit_labels = String[]
        for q in 1:n_vars
            bit = 1 << (q - 1)
            has_x = (pauli.x & bit) != 0
            has_z = (pauli.z & bit) != 0
            if has_x && has_z
                push!(qubit_labels, "Y$q")
            elseif has_x
                push!(qubit_labels, "X$q")
            elseif has_z
                push!(qubit_labels, "Z$q")
            end
        end
        push!(term_strings, isempty(qubit_labels) ? "I" : join(qubit_labels))
    end
    return join(term_strings, "+")
end
