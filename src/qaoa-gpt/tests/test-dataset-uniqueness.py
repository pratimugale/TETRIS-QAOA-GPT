import pandas as pd
import sys
import ast
from tqdm import tqdm

def make_canonical(formula):
    """
    Sorts a formula (list of clauses) to handle logical equivalence.
    Assumes formula is a list of lists, e.g., [[1, 2, 0.5], [3, 4, 1.0]].
    """
    if not isinstance(formula, list):
        return formula
    
    # 1. Sort literals within each clause (e.g., [2, 1] -> [1, 2])
    # 2. Sort the list of clauses themselves
    # Note: Using str() inside sorted to handle mixed types if any
    sorted_clauses = sorted([tuple(sorted(clause)) for clause in formula])
    
    # 3. Return as a hashable tuple of tuples
    return tuple(sorted_clauses)

def main():
    if len(sys.argv) != 2:
        print("Usage: python src/qaoa-gpt/tests/test-dataset-uniqueness.py <csv_path>")
        sys.exit(1), 

    csv_path = sys.argv[1]
    print(f"Loading dataset from: {csv_path} ...")
    df = pd.read_csv(csv_path)

    print(f"\n[1/3] Checking raw row uniqueness ...")
    total_rows = len(df)
    unique_rows = df.drop_duplicates().shape[0]
    print(f"Total Rows:        {total_rows}")
    print(f"Unique Raw Rows:   {unique_rows}")
    if total_rows != unique_rows:
        print(f"Warning: Found {total_rows - unique_rows} identical rows in the CSV.")

    print(f"\n[2/3] Parsing and canonicalizing formulae ...")
    # Convert string representation to Python list
    tqdm.pandas()
    df['parsed_formula'] = df['formula_list'].progress_apply(ast.literal_eval)
    
    # Create canonical hashable representation
    df['formula_canonical'] = df['parsed_formula'].apply(make_canonical)

    print(f"\n[3/3] Final Uniqueness Report")
    print("-" * 40)
    n_unique_formulae = df['formula_canonical'].nunique()
    print(f"Total rows in dataset:       {total_rows}")
    print(f"Unique canonical formulae:   {n_unique_formulae}")
    print(f"Duplicate formulae found:    {total_rows - n_unique_formulae}")
    print("-" * 40)

    if n_unique_formulae == total_rows:
        print("Success: All formulae are logically unique (even after sorting).")
    else:
        duplicate_percentage = (total_rows - n_unique_formulae) / total_rows * 100
        print(f"Redundancy Detected: {duplicate_percentage:.2f}% of your dataset contains duplicate problems.")
        
        # Optional: Show some duplicates
        # dups = df[df.duplicated('formula_canonical', keep=False)].sort_values('formula_canonical')
        # print("\nFirst 5 duplicates (Sample):")
        # print(dups[['filename', 'n_layers']].head(10))

if __name__ == "__main__":
    main()