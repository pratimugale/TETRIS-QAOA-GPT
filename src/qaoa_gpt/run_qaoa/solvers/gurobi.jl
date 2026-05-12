# This file implements the exact solver for MaxE3SAT using Gurobi.
# It translates the MaxE3SAT problem into a MILP problem and solves it using Gurobi.

using PyCall
gp = pyimport("gurobipy")

"""
    solve_max_e3sat_exact(n_vars, clauses)

Solves the MaxE3SAT problem exactly using Gurobi.
Returns the maximum number of satisfied clauses and the bitstring of the ground state.
"""
function solve_max_e3sat_exact(n_vars, clauses)
    model = gp.Model("MaxE3SAT_GroundState")
    model.setParam("OutputFlag", 0)

    # 1. Decision variables
    x = model.addVars(n_vars, vtype=gp.GRB.BINARY, name="x")

    # 2. Objective Function: Minimize Unsatisfied Clauses
    # In QAOA, this is your Cost Hamiltonian H_C
    cost_expr = gp.LinExpr()

    for clause in clauses
        # A clause is UNSATISFIED only if all literals are false.
        # We create a variable that is 1 ONLY if the clause is broken.
        is_broken = model.addVar(vtype=gp.GRB.BINARY)

        clause_literals = []
        for lit in clause
            var_idx = abs(lit) - 1
            # If literal is negative, it's true when x is 0
            # If literal is positive, it's true when x is 1
            push!(clause_literals, lit > 0 ? x[var_idx] : (1 - x[var_idx]))
        end

        # Constraint: is_broken is 1 only if all literals are 0
        # This is the "AND" of the negations
        model.addConstr(gp.quicksum(clause_literals) >= (1 - is_broken))
        model.addGenConstrIndicator(is_broken, 1, gp.quicksum(clause_literals) == 0)

        cost_expr += is_broken
    end

    # Minimize cost (unsatisfied clauses)
    model.setObjective(cost_expr, gp.GRB.MINIMIZE)
    model.optimize()

    if model.Status == gp.GRB.OPTIMAL
        ground_state = [round(Int, x[i].X) for i in 0:n_vars-1]
        unsatisfied = model.ObjVal
        return (length(clauses) - unsatisfied), ground_state
    end
end
