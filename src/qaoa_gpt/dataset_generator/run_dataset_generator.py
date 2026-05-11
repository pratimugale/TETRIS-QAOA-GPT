import argparse
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.qaoa_gpt.dataset_generator.generator import DatasetGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate large-scale stratified Max-3SAT dataset.")
    parser.add_argument("--n_vars", type=int, default=10, help="Number of variables")
    parser.add_argument("--total", type=int, default=100, help="Total target number of instances")
    parser.add_argument("--seed", type=int, default=2000, help="Base seed")
    parser.add_argument("--split", action="store_true", help="Create train/test splits after generation")
    
    args = parser.parse_args()
    
    # We want half random, half balanced.
    # Within each, half sat, half unsat.
    # So sat_goal = total / 4, unsat_goal = total / 4
    # The DatasetGenerator by default generates the required number of instances
    # for both random and balanced Max-3SAT instances.
    goal_per_bucket = args.total // 4
    
    gen = DatasetGenerator(
        n_vars=args.n_vars,
        sat_goal_per_type=goal_per_bucket,
        unsat_goal_per_type=goal_per_bucket,
        base_seed=args.seed
    )
    
    gen.run_pipeline()
    
    # current Makefile won't pass --split for the train/val/test split here, 
    # because we will reuse the logic in prepare_circ.py for the split
    if args.split:
        gen.create_splits()

if __name__ == "__main__":
    main()
