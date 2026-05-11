import os
import csv
import json
import hashlib
import random
import shutil
from pathlib import Path
from satqubolib.generators import BalancedSAT
from src.solvers.gurobi import solve_max_e3sat_exact

class DatasetGenerator:
    def __init__(self, n_vars, sat_goal_per_type, unsat_goal_per_type, base_seed=2000, output_root="dataset"):
        self.n_vars = n_vars
        self.sat_goal = sat_goal_per_type
        self.unsat_goal = unsat_goal_per_type
        self.base_seed = base_seed
        self.output_root = Path(output_root) / f"n{n_vars}"
        
        # Define Critical Phase Transition Ratios
        self.ratios = {
            "balanced": 3.6,
            "random": 4.26
        }
        
        # Directories
        self.dirs = {
            "all": self.output_root / "all",
            "sat": self.output_root / "sat",
            "unsat": self.output_root / "unsat"
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
            
        # Registry and Metadata
        self.metadata_path = self.output_root / "metadata.csv"
        self.seen_hashes = set()
        self.load_registry()
        
    def load_registry(self):
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.seen_hashes.add(row['hash'])

    # Format: (variable index, 0 if positive literal else 1)
    def _literal_key(self, lit):
        var = abs(lit)
        return (var, 0 if lit > 0 else 1)

    def canonicalize_formula(self, clauses):
        # 1. Sort literals within each clause
        sorted_clauses = [sorted(c, key=self._literal_key) for c in clauses]
        
        # 2. Sort clauses within formula
        def clause_key(c):
            return tuple(self._literal_key(lit) for lit in c)
        
        sorted_clauses.sort(key=clause_key)
        
        # 3. Generate human-readable string for the canonical formula map
        clause_strs = []
        for c in sorted_clauses:
            lit_strs = []
            for lit in c:
                if lit > 0:
                    lit_strs.append(f"x{lit}")
                else:
                    lit_strs.append(f"~x{-lit}")
            clause_strs.append("(" + " v ".join(lit_strs) + ")")
            
        return sorted_clauses, " ^ ".join(clause_strs)

    def get_hash(self, canonical_formula_str):
        return hashlib.sha256(canonical_formula_str.encode()).hexdigest()

    def generate_unforced_instance(self, type_name, seed):
        """Generates a random instance without a forced solution."""
        random.seed(seed)
        
        # Apply +/- 20% random variation to the target ratio
        base_ratio = self.ratios[type_name]
        variation = (random.random() * 0.4) - 0.2
        instance_ratio = base_ratio * (1.0 + variation)
        num_clauses = int(self.n_vars * instance_ratio)
        
        clauses = []

        # clause is a tuple of literals
        seen_clauses = set()
        
        if type_name == "balanced":
            generator = BalancedSAT(self.n_vars, num_clauses, vars_per_clause=3)
            raw_formula = generator.generate()
            
            for c in raw_formula.clauses:
                can_clause = tuple(sorted(c, key=self._literal_key))
                if can_clause not in seen_clauses:
                    seen_clauses.add(can_clause)
                    clauses.append(list(c))
            
            return clauses
        else:
            # Standard Uniform Random sampling
            while len(clauses) < num_clauses:
                vars_idx = random.sample(range(1, self.n_vars + 1), 3) # pick 3 distinct variables
                clause = [v if random.random() < 0.5 else -v for v in vars_idx]
                
                # Ensure that clause is unique
                can_clause = tuple(sorted(clause, key=self._literal_key))
                
                if can_clause not in seen_clauses:
                    seen_clauses.add(can_clause)
                    clauses.append(clause)
        
        return clauses

    # run the pipeline to generate the dataset
    def run_pipeline(self):
        print(f"Starting pipeline for N={self.n_vars}")
        print(f"Goal per type: {self.sat_goal} SAT, {self.unsat_goal} UNSAT")
        
        stats = {
            "balanced": {"sat": 0, "unsat": 0},
            "random": {"sat": 0, "unsat": 0}
        }
        
        # Load existing stats from metadata if it exists
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t = row.get('formula_method', row.get('type', 'random'))
                    s = 'sat' if row['is_sat'] == 'True' else 'unsat'
                    if t in stats:
                        stats[t][s] += 1
        
        header = ["formula_num", "filename", "n_vars", "n_clauses", "formula_method", "is_sat", "max_satisfied", "hash", "formula_list"]
        file_exists = self.metadata_path.exists()
        
        # Determine starting formula_num
        start_num = 0
        if file_exists:
            with open(self.metadata_path, 'r') as f:
                start_num = sum(1 for _ in f) - 1 # exclude header

        with open(self.metadata_path, "a", newline="") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=header)
            if not file_exists:
                writer.writeheader()

            # Global iteration counter for seeds
            offset = len(self.seen_hashes)
            
            while any(stats[t]["sat"] < self.sat_goal or stats[t]["unsat"] < self.unsat_goal for t in ["balanced", "random"]):
                for t in ["balanced", "random"]:
                    # Skip if this type is fully satisfied
                    if stats[t]["sat"] >= self.sat_goal and stats[t]["unsat"] >= self.unsat_goal:
                        continue
                        
                    offset += 1
                    clauses = self.generate_unforced_instance(t, self.base_seed + offset)
                    
                    sorted_clauses, can_formula = self.canonicalize_formula(clauses)
                    f_hash = self.get_hash(can_formula)
                    
                    if f_hash in self.seen_hashes:
                        continue
                    
                    self.seen_hashes.add(f_hash)
                    
                    # Solve with Gurobi
                    max_sat, ground_state = solve_max_e3sat_exact(self.n_vars, clauses)
                    is_sat = (max_sat == len(clauses))
                    label = "sat" if is_sat else "unsat"
                    
                    # Stratification check: if we already have enough of this bucket, discard and continue
                    if stats[t][label] >= (self.sat_goal if is_sat else self.unsat_goal):
                        continue
                    
                    filename = f"{t}_v{self.n_vars}_c{len(clauses)}_{f_hash[:8]}.cnf"
                    temp_path = self.dirs["all"] / filename
                    final_path = self.dirs[label] / filename
                    
                    # Write DIMACS
                    with open(temp_path, "w") as f_dimacs:
                        f_dimacs.write(f"p cnf {self.n_vars} {len(clauses)}\n")
                        for c in sorted_clauses:
                            f_dimacs.write(f"{c[0]} {c[1]} {c[2]} 0\n")
                    
                    # Move to bucket
                    shutil.move(str(temp_path), str(final_path))
                    
                    # Record
                    start_num += 1
                    writer.writerow({
                        "formula_num": start_num,
                        "filename": filename,
                        "n_vars": self.n_vars,
                        "n_clauses": len(clauses),
                        "formula_method": t,
                        "is_sat": is_sat,
                        "max_satisfied": max_sat,
                        "hash": f_hash,
                        "formula_list": json.dumps(sorted_clauses)
                    })
                    f_csv.flush()
                    self.seen_hashes.add(f_hash)
                    
                    stats[t][label] += 1
                    print(f"[{t.upper()} {label.upper()}] {stats[t]['sat']}/{self.sat_goal} SAT, {stats[t]['unsat']}/{self.unsat_goal} UNSAT")

    # We end up not using this function, and instead split the data in prepare_circ.py once all MosaicADAPT-QAOA 
    # circuits are prepared for all instances
    def create_splits(self, train_ratio=0.8, val_ratio=0.1):
        print("Creating Train/Val/Test splits...")
        train_dir = self.output_root / "train"
        val_dir = self.output_root / "val"
        test_dir = self.output_root / "test"
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)
        
        # We split within each bucket (SAT/UNSAT x Balanced/Random)
        buckets = [
            (self.dirs["sat"], "balanced"),
            (self.dirs["sat"], "random"),
            (self.dirs["unsat"], "balanced"),
            (self.dirs["unsat"], "random")
        ]
        
        for base_dir, type_name in buckets:
            if not base_dir.exists():
                continue
            files = [f for f in os.listdir(base_dir) if f.startswith(type_name)]
            random.seed(42) # Deterministic split
            random.shuffle(files)
            
            train_idx = int(len(files) * train_ratio)
            val_idx = train_idx + int(len(files) * val_ratio)
            
            train_files = files[:train_idx]
            val_files = files[train_idx:val_idx]
            test_files = files[val_idx:]
            
            for f in train_files:
                shutil.move(str(base_dir / f), str(train_dir / f))
            for f in val_files:
                shutil.move(str(base_dir / f), str(val_dir / f))
            for f in test_files:
                shutil.move(str(base_dir / f), str(test_dir / f))
        
        # Cleanup empty bucket directories
        shutil.rmtree(self.dirs["sat"], ignore_errors=True)
        shutil.rmtree(self.dirs["unsat"], ignore_errors=True)
        shutil.rmtree(self.dirs["all"], ignore_errors=True)
        
        # Update metadata.csv with split info
        if self.metadata_path.exists():
            print("Mapping splits back to metadata.csv...")
            split_map = {}
            for split_name, directory in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
                for fname in os.listdir(directory):
                    split_map[fname] = split_name
                    
            with open(self.metadata_path, 'r') as f:
                reader = list(csv.DictReader(f))
                
            if reader:
                fieldnames = list(reader[0].keys())
                if "split" not in fieldnames:
                    fieldnames.append("split")
                    
                with open(self.metadata_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in reader:
                        fname = row.get("filename")
                        row["split"] = split_map.get(fname, "unknown")
                        writer.writerow(row)
                
        print(f"Splits complete. Train: {len(os.listdir(train_dir))}, Val: {len(os.listdir(val_dir))}, Test: {len(os.listdir(test_dir))}")
