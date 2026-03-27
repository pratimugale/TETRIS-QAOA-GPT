import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import networkx as nx
import ast
from lcg_feather import SATGraphEmbedder

def generate_random_3sat(n_vars, n_clauses):
    formula = []
    vars_list = list(range(1, n_vars + 1))
    for _ in range(n_clauses):
        clause_vars = random.sample(vars_list, 3)
        clause = [v if random.random() > 0.5 else -v for v in clause_vars]
        formula.append(clause)
    return formula

def create_isomorphic_formula(formula, n_vars):
    p = list(range(1, n_vars + 1))
    random.shuffle(p)
    mapping = {original: new for original, new in zip(range(1, n_vars + 1), p)}
    new_formula = []
    for clause in formula:
        new_clause = [mapping[abs(lit)] * (1 if lit > 0 else -1) for lit in clause]
        random.shuffle(new_clause)
        new_formula.append(new_clause)
    random.shuffle(new_formula)
    return new_formula

def create_editN_formula(formula, n_vars, n_edits):
    new_formula = [list(c) for c in formula]
    for _ in range(n_edits):
        c_idx = random.randint(0, len(new_formula) - 1)
        l_idx = random.randint(0, 2)
        current_lit = new_formula[c_idx][l_idx]
        new_var = random.randint(1, n_vars)
        new_lit = new_var if random.random() > 0.5 else -new_var
        # Try to find a different literal, but don't loop forever if formula is small
        new_formula[c_idx][l_idx] = new_lit
    return new_formula

def main():
    n_vars = 12
    n_clauses = int(round(4.26 * n_vars)) # 51
    
    embedder = SATGraphEmbedder(n_nodes=n_vars, negation_weight=1.0)
    
    print(f"Generating base formula (N={n_vars}, M={n_clauses})...")
    base_formula = generate_random_3sat(n_vars, n_clauses)
    
    print("Generating 100 isomorphic variants...")
    iso_formulas = [create_isomorphic_formula(base_formula, n_vars) for _ in range(100)]
    
    print("Generating 100 'Edit-1' variants...")
    edit1_formulas = [create_editN_formula(base_formula, n_vars, 1) for _ in range(100)]
    
    print("Generating 100 'Edit-4' variants...")
    edit4_formulas = [create_editN_formula(base_formula, n_vars, 4) for _ in range(100)]
    
    print("Generating 100 'Edit-10' variants...")
    edit10_formulas = [create_editN_formula(base_formula, n_vars, 10) for _ in range(100)]
    
    print("Generating 100 random different formulas...")
    random_formulas = [generate_random_3sat(n_vars, n_clauses) for _ in range(100)]
    
    all_formulas = [base_formula] + iso_formulas + edit1_formulas + edit4_formulas + edit10_formulas + random_formulas
    labels = ["Base"] + ["Isomorphic"] * 100 + ["Edit-1"] * 100 + ["Edit-4"] * 100 + ["Edit-10"] * 100 + ["Random"] * 100
    colors = ["red"] + ["blue"] * 100 + ["orange"] * 100 + ["purple"] * 100 + ["cyan"] * 100 + ["green"] * 100
    
    print(f"Total formulas to embed: {len(all_formulas)}")
    embs = embedder.get_embeddings(all_formulas)
    
    # Use only original groups for T-SNE/PCA to keep them readable
    viz_mask = [l in ["Base", "Isomorphic", "Edit-1", "Random"] for l in labels]
    viz_embs = embs[viz_mask]
    viz_labels = [l for l in labels if l in ["Base", "Isomorphic", "Edit-1", "Random"]]
    viz_colors = [c for i, c in enumerate(colors) if viz_mask[i]]
    
    print(f"Running T-SNE (perplexity=30) and PCA on visualization subset...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    embs_tsne = tsne.fit_transform(viz_embs)
    
    pca = PCA(n_components=2)
    embs_pca = pca.fit_transform(viz_embs)
    
    # Use raw coordinates
    embs_2d_final = embs_tsne

    print("Plotting results...")
    plt.figure(figsize=(10, 8))
    
    unique_labels = ["Random", "Edit-1", "Isomorphic", "Base"] # Plot Base last to be on top
    for label in unique_labels:
        idx = [j for j, l in enumerate(viz_labels) if l == label]
        c = viz_colors[idx[0]]
        plt.scatter(embs_2d_final[idx, 0], embs_2d_final[idx, 1], c=c, label=label, s=100, alpha=0.7)
        
    plt.legend()
    plt.title(f"Sensitivity Test: T-SNE (Perplexity=30) (N={n_vars}, M={n_clauses})")
    plt.xlabel("TSNE Dimension 1")
    plt.ylabel("TSNE Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path_tsne = "sat_sensitivity_tsne.png"
    plt.savefig(save_path_tsne)
    print(f"T-SNE plot saved to {save_path_tsne}")

    # Plot PCA
    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        idx = [j for j, l in enumerate(viz_labels) if l == label]
        c = viz_colors[idx[0]]
        plt.scatter(embs_pca[idx, 0], embs_pca[idx, 1], c=c, label=label, s=100, alpha=0.7)
    plt.legend()
    plt.title(f"Sensitivity Test: PCA (Linear Projection) (N={n_vars}, M={n_clauses})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path_pca = "sat_sensitivity_pca.png"
    plt.savefig(save_path_pca)
    print(f"PCA plot saved to {save_path_pca}")
    
    # Euclidean distance stats
    base_emb = embs[0]
    all_dists = []
    group_labels = ["Isomorphic", "Edit-1", "Edit-4", "Edit-10", "Random"]
    
    print(f"\nDistances from Base:")
    curr_idx = 1
    for group_name in group_labels:
        start = curr_idx
        end = curr_idx + 100
        group_embs = embs[start:end]
        dists = [np.linalg.norm(base_emb - e) for e in group_embs]
        all_dists.append(dists)
        print(f"  Avg {group_name:10} Distance: {np.mean(dists):.6f}")
        curr_idx = end

    # Generate Box Plot
    plt.figure(figsize=(10, 8))
    plt.boxplot(all_dists, labels=group_labels, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red'))
    
    plt.title(f"Distance from Base Formula (Euclidean) (N={n_vars}, M={n_clauses})")
    plt.ylabel("Euclidean Distance")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path_box = "sat_sensitivity_boxplot.png"
    plt.savefig(save_path_box)
    print(f"Box plot saved to {save_path_box}")

if __name__ == "__main__":
    main()
