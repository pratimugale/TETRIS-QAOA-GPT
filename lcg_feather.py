import networkx as nx
import numpy as np
import math
import pandas as pd
import ast
import json
from tqdm import tqdm
from FEATHER.src.feather import FEATHERG

class SATGraphEmbedder:
    """
    A utility class to generate graph embeddings for Max-3-SAT formulas
    using Literal-Clause Graphs (LCG) and the FEATHER algorithm.
    """
    def __init__(self, n_nodes, theta_max=2.5, eval_points=25, order=5, pooling="mean", negation_weight=1.0):
        self.n_nodes = n_nodes
        self.negation_weight = negation_weight
        self.feather = FEATHERG(
            theta_max=theta_max, 
            eval_points=eval_points, 
            order=order, 
            pooling=pooling
        )

    def formula_to_lcg(self, formula_list):
        """
        Converts a SAT formula into a Literal-Clause Graph (LCG).
        
        Node Mapping (consistent with FEATHER requirement of 0..N-1 indexing):
        - Literals x_1..x_n:  0 to n_nodes-1
        - Literals -x_1..-x_n: n_nodes to 2*n_nodes-1
        - Clauses C_1..C_m:   2*n_nodes to 2*n_nodes + m - 1
        
        Edges:
        1. Literal-Clause: Edge between literal and clause it appears in.
        2. Negation (Contradiction): Edge between x_i and -x_i.
        """
        n = self.n_nodes
        m = len(formula_list)
        G = nx.Graph()
        
        # Total nodes: 2*literals + m clauses
        total_nodes = 2 * n + m
        G.add_nodes_from(range(total_nodes))
        
        # 1. Add Negation Edges (x_i <-> -x_i)
        # We use negation_weight to distinguish these from literal-clause edges.
        for i in range(n):
            pos_lit_node = i
            neg_lit_node = i + n
            G.add_edge(pos_lit_node, neg_lit_node, weight=self.negation_weight)
            
        # 2. Add Clause Edges
        for j, clause in enumerate(formula_list):
            clause_node = 2 * n + j
            for lit in clause:
                if lit > 0:
                    # Positive literal: x_1 maps to 0, x_n maps to n-1
                    lit_node = lit - 1
                else:
                    # Negative literal: -x_1 maps to n, -x_n maps to 2n-1
                    lit_node = abs(lit) - 1 + n
                
                # Boundary check (optional but safe)
                if lit_node < 2 * n:
                    G.add_edge(lit_node, clause_node, weight=1.0)
                    
        return G

    def get_embeddings(self, formula_list_of_lists):
        """
        Generates embeddings for a batch of formulas.
        Args:
            formula_list_of_lists: List of formulas (each formula is a list of lists of literals)
        Returns:
            numpy.ndarray: (n_formulas, 500) embedding matrix.
        """
        graphs = []
        for formula in tqdm(formula_list_of_lists, desc="Converting formulas to LCGs"):
            graphs.append(self.formula_to_lcg(formula))
        
        print(f"Running FEATHER fitting on {len(graphs)} graphs...")
        self.feather.fit(graphs)
        return self.feather.get_embedding()

if __name__ == "__main__":
    # Standalone Demo/Test
    n_nodes = 12
    embedder = SATGraphEmbedder(n_nodes=n_nodes)
    
    # 1. Create dummy formula (3-SAT)
    # (x1 or x2 or -x3) AND (-x1 or x4 or x5)
    sample_formula = [[1, 2, -3], [-1, 4, 5]]
    
    print("Testing single formula to LCG conversion...")
    G = embedder.formula_to_lcg(sample_formula)
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    # Expected nodes: 2*12 + 2 = 26
    # Expected edges: 12 (negation) + 6 (clause literals) = 18
    
    # 2. Test embedding generation
    print("\nGenerating embedding for sample formula batch...")
    # Wrap in list to simulate batch
    embs = embedder.get_embeddings([sample_formula])
    print(f"Embedding shape: {embs.shape}")
    print(f"First 10 values: {embs[0][:10]}")
    
    # 3. Test on real data if possible
    csv_path = "/Users/pratim/work/thesis/TETRIS-ADAPT-QAOA-GPT/results/n12/adapt_qaoa_5layer_standardqaoamixer/optimized_circuits.csv"
    try:
        print(f"\nAttempting to load real formula from: {csv_path}")
        df = pd.read_csv(csv_path, nrows=5)
        # Parse the string representation of list-of-lists
        real_formulas = [ast.literal_eval(f) for f in df['formula_list']]
        
        real_embs = embedder.get_embeddings(real_formulas)
        print(f"Real data embedding shape: {real_embs.shape}")
        
    except FileNotFoundError:
        print("CSV not found, skipping real data test.")
    except Exception as e:
        print(f"Error during real data test: {e}")
