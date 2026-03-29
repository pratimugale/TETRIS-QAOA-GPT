import networkx as nx
from tqdm import tqdm
import sys
from pathlib import Path
# Add project root to sys.path so FEATHER directory can be imported as a package
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from FEATHER.src.feather import FEATHERG

class SATGraphEmbedder:
    """
    A utility class to generate graph embeddings for Max-3-SAT formulas
    using Literal-Clause Graphs (LCG) and the FEATHER algorithm.
    """
    def __init__(self, n_nodes, theta_max=2.5, eval_points=25, order=5, pooling="mean"):
        self.n_nodes = n_nodes
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

        """
        n = self.n_nodes
        m = len(formula_list)
        G = nx.Graph()
        
        # Total nodes: 2*literals + m clauses
        total_nodes = 2 * n + m
        G.add_nodes_from(range(total_nodes))
        
        # 1. Add Clause Edges

        for j, clause in enumerate(formula_list):
            clause_node = 2 * n + j
            for lit in clause:
                if lit > 0:
                    # Positive literal: x_1 maps to 0, x_n maps to n-1
                    lit_node = lit - 1
                else:
                    # Negative literal: -x_1 maps to n, -x_n maps to 2n-1
                    lit_node = abs(lit) - 1 + n
                
                # Boundary check
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
