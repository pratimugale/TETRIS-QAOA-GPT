import gurobipy as gp
from gurobipy import GRB

def solve_max_e3sat_exact(n_vars, clauses):
    """
    Solves the Max-3SAT problem exactly using Gurobi.
    Returns:
        max_satisfied (int): Maximum number of satisfied clauses.
        ground_state (list): Bitstring (list of 0/1) for the ground state.
    """
    model = gp.Model("MaxE3SAT_GroundState")
    model.setParam("OutputFlag", 0)

    # 1. Decision variables
    x = model.addVars(n_vars, vtype=GRB.BINARY, name="x")

    # 2. Objective Function: Minimize Unsatisfied Clauses
    cost_expr = gp.LinExpr()

    for clause in clauses:
        # A clause is UNSATISFIED only if all literals are false.
        # i.e., literal satisfies when x is 1 (if positive) or 0 (if negative).
        # We create a variable that is 1 ONLY if the clause is broken.
        is_broken = model.addVar(vtype=GRB.BINARY)

        clause_literals = []
        for lit in clause:
            var_idx = abs(lit) - 1
            # If literal is negative, it's true when x is 0 -> (1 - x)
            # If literal is positive, it's true when x is 1 -> x
            clause_literals.append(x[var_idx] if lit > 0 else (1 - x[var_idx]))

        # Constraint: is_broken is 1 only if all literals are 0
        # This is equivalent to: sum(clause_literals) >= 1 - is_broken
        # And: if is_broken=1, then sum(clause_literals) == 0 (handled by indicator)
        model.addConstr(gp.quicksum(clause_literals) >= (1 - is_broken))
        model.addGenConstrIndicator(is_broken, 1, gp.quicksum(clause_literals) == 0)

        cost_expr += is_broken

    # Minimize cost (unsatisfied clauses)
    model.setObjective(cost_expr, GRB.MINIMIZE)
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        ground_state = [int(round(x[i].X)) for i in range(n_vars)]
        unsatisfied = int(round(model.ObjVal))
        return (len(clauses) - unsatisfied), ground_state
    else:
        raise Exception(f"Gurobi failed to find an optimal solution. Status: {model.Status}")
