import pandas as pd
import numpy as np
import argparse
import os

def summarize_results(csv_path):
    print(f"Loading results from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Calculate additional flags
    # Formula Failure: best_ar is 0.0 (all samples failed)
    df['is_formula_failure'] = (df['best_ar'] == 0.0)
    
    # Stratify by type
    types = df['type'].unique()
    
    summary_rows = []
    
    def process_group(group_df, label):
        n_formulas = len(group_df)
        n_formulas_valid = (group_df['best_ar'] > 0).sum()
        
        # 1. Failure Rates
        # Formula Failure Rate: Percentage of formulas where zero samples worked
        formula_fail_rate = (group_df['is_formula_failure'].sum() / n_formulas) * 100
        
        # Circuit Failure Rate: Average of error_rate column
        # error_rate in CSV is the fraction of invalid circuits for THAT formula, i.e. 
        # the AR for the sampled circuits is already averaged, which we average over all formulas
        circuit_fail_rate = group_df['error_rate'].mean() * 100
        
        # 2. AR Metrics
        # AR (Valid): Mean for formulas that had at least one valid circuit
        if n_formulas_valid > 0:
            best_ar_valid = group_df[group_df['best_ar'] > 0]['best_ar'].mean() * 100
            # at a formula level, the avg_ar is already the average of only the valid samples. 
            # see src/qaoa-gpt/adapt_gpt_eval_energy.jl
            avg_ar_valid = group_df[group_df['best_ar'] > 0]['avg_ar'].mean() * 100
        else:
            best_ar_valid, avg_ar_valid = 0.0, 0.0
        
        # 3. Ground Truth Ref
        avg_ar_adapt = group_df['ar_adapt'].mean() * 100
        
        return {
            'Category': label.upper(),
            'Count': n_formulas,
            'Formula Error Rate (Best ER)': round(formula_fail_rate, 2),
            'Circuit Error Rate (Avg ER)': round(circuit_fail_rate, 2),
            'Best AR GPT (Valid Circuits)': round(best_ar_valid, 2),
            'Average AR GPT (Valid Circuits)': round(avg_ar_valid, 2),
            'MosaicA-QAOA': round(avg_ar_adapt, 2)
        }

    # Process each type
    for t in sorted(types):
        summary_rows.append(process_group(df[df['type'] == t], t))
        
    # Process Global
    summary_rows.append(process_group(df, "Global Summary"))
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save output
    output_path = os.path.join(os.path.dirname(csv_path), "stratified_stats.csv")
    summary_df.to_csv(output_path, index=False)
    
    print("\n--- Summary Statistics ---")
    print(summary_df.to_string(index=False))
    print(f"\nStats saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to testset_summary.csv")
    args = parser.parse_args()
    
    summarize_results(args.input)
