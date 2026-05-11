# Tokenizer for Q3SAT - only the input formulas (not the QAOA circuits)
def tokenize_formula(formula_list):
    tokens_seq_list = []
    for clause in formula_list:
        for lit in clause:
            if lit > 0:
                tokens_seq_list.append(f"x{lit}")
            else:
                tokens_seq_list.append(f"~x{abs(lit)}")
        tokens_seq_list.append('|')
    tokens_seq_list.append('end_of_formula')
    return tokens_seq_list

# TODO: add constrained decoding here in the future
