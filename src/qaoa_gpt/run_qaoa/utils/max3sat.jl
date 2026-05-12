
"""
    convert_dict_to_formula(formula_dict::Dict)::ADAPT.Hamiltonians.Max3SAT.Types.Formula
Convert JSON instance data to Max3SAT formula format.
"""
function get_formula_as_struct(formula_data)::ADAPT.Hamiltonians.Max3SAT.Types.Formula
    clauses = ADAPT.Hamiltonians.Max3SAT.Types.Clause[]

    for clause_data in formula_data
        literals = ADAPT.Hamiltonians.Max3SAT.Types.Literal[]
        for lit_data in clause_data
            lit = ADAPT.Hamiltonians.Max3SAT.Types.Literal(lit_data["var"], lit_data["neg"])
            push!(literals, lit)
        end
        clause = ADAPT.Hamiltonians.Max3SAT.Types.Clause((literals[1], literals[2], literals[3]))
        push!(clauses, clause)
    end

    return ADAPT.Hamiltonians.Max3SAT.Types.Formula(clauses)
end

"""
    get_formula_as_list(formula_data)::Vector{Vector{Int}}
Convert JSON instance data to a raw Vector of Vector of literals (Integers).
Used for the Gurobi exact solver.
"""
function get_formula_as_list(formula_data)
    clauses = Vector{Int}[]
    for clause_data in formula_data
        literals = Int[]
        for lit_data in clause_data
            # lit_data["neg"] = true means it is -var
            lit = lit_data["neg"] ? -lit_data["var"] : lit_data["var"]
            push!(literals, lit)
        end
        push!(clauses, literals)
    end
    return clauses
end

"""
    get_number_of_satisfied_clauses(bitstring, formula)
Evaluate how many clauses are satisfied by a given bitstring.
Returns the number of clauses satisfied by the bitstring.
"""
function get_number_of_satisfied_clauses(bitstring::Vector{Bool}, formula::ADAPT.Hamiltonians.Max3SAT.Types.Formula)::Int
    number_of_satisfied_clauses = 0
    for clause in formula.clauses
        clause_satisfied = any(lit.neg ? !bitstring[lit.var] : bitstring[lit.var] for lit in clause.lits)
        number_of_satisfied_clauses += clause_satisfied
    end
    return number_of_satisfied_clauses
end

"""
    get_best_bitstring_among_sampled_bitstrings(sampled_bitstrings::Vector{Vector{Bool}}, 
    formula::ADAPT.Hamiltonians.Max3SAT.Types.Formula)::Tuple{Vector{Bool}, Int}
Find the best bitstring among the sampled bitstrings.
Returns the best bitstring and the number of clauses satisfied by the best bitstring.
"""
function get_best_bitstring_among_sampled_bitstrings(sampled_bitstrings::Vector{Vector{Bool}},
    formula::ADAPT.Hamiltonians.Max3SAT.Types.Formula)::Tuple{Vector{Bool},Int}

    best_satisfaction_count = -1
    best_bitstring = nothing

    for bitstring in sampled_bitstrings
        satisfaction_count = get_number_of_satisfied_clauses(bitstring, formula)
        if satisfaction_count > best_satisfaction_count
            best_satisfaction_count = satisfaction_count
            best_bitstring = bitstring
        end
    end

    return best_bitstring, best_satisfaction_count
end

"""
    parse_cnf_file(cnf_file_path::String)::Dict
Parse a single CNF file and return a dictionary with the same structure as JSON instances.
Returns: Dict with "variables", "instance_id", and "formula" keys.
"""
function parse_cnf_file(cnf_file_path::String)::Dict
    open(cnf_file_path, "r") do f
        num_vars = 0
        num_clauses = 0
        clauses = []

        for line in eachline(f)
            stripped_line = strip(line)

            # Skip empty lines, comment lines (starting with 'c'), and separator lines (starting with '%')
            if isempty(stripped_line) || startswith(stripped_line, "c") || startswith(stripped_line, "%")
                continue
            end

            # Parse header: p cnf <num_vars> <num_clauses>
            if startswith(stripped_line, "p")
                parts = split(stripped_line)
                if length(parts) >= 4 && parts[1] == "p" && parts[2] == "cnf"
                    num_vars = parse(Int, parts[3])
                    num_clauses = parse(Int, parts[4])
                end
                continue
            end

            # Parse clause line (ends with 0)
            clause_literals = []
            for token in split(stripped_line)
                # Skip if token is '%' (separator character)
                if token == "%"
                    break
                end
                lit = parse(Int, token)
                if lit == 0
                    # End of clause
                    if length(clause_literals) == 3
                        # Convert to JSON format: array of objects with "var" and "neg"
                        clause_data = []
                        for literal in clause_literals
                            var = abs(literal)
                            neg = literal < 0
                            push!(clause_data, Dict("var" => var, "neg" => neg))
                        end
                        push!(clauses, clause_data)
                    end
                    break
                else
                    push!(clause_literals, lit)
                end
            end
        end

        # Extract instance ID from filename (e.g., "uf20-01.cnf" -> 1)
        filename = basename(cnf_file_path)
        instance_id_match = match(r"uf\d+-(\d+)\.cnf", filename)
        instance_id = instance_id_match !== nothing ? parse(Int, instance_id_match.captures[1]) : 1

        return Dict(
            "variables" => num_vars,
            "instance_id" => instance_id,
            "formula" => clauses
        )
    end
end

"""
    load_dataset(dataset_path::String)::Union{Dict, String}
Load the Max3SAT dataset from either a JSON file or a directory of CNF files.
For JSON files, returns the parsed dataset.
For CNF directories, returns the directory path (lazy loading - files will be processed one at a time).
"""
function load_dataset(dataset_path::String)::Union{Dict,String}
    println("Loading dataset from $dataset_path...")

    if !isfile(dataset_path) && !isdir(dataset_path)
        error("Dataset path not found: $dataset_path")
    end

    if isdir(dataset_path)
        # For CNF directories, return the path for lazy loading
        cnf_files = filter(f -> endswith(f, ".cnf"), readdir(dataset_path, join=false))
        println("Found $(length(cnf_files)) CNF files (will be loaded one at a time)")
        return dataset_path  # Return path instead of loading everything
    else
        # Load from JSON file (existing behavior)
        dataset = JSON.parsefile(dataset_path)
        println("Dataset loaded successfully!")
        return dataset
    end
end
