import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import matplotlib

# Set global styles for paper-ready legibility
matplotlib.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze variance of predicted circuit parameters.')
    parser.add_argument('--input', type=str, required=True, help='Path to testset_eval_output.json')
    parser.add_argument('--out_dir', type=str, default='analysis_plots', help='Output directory for plots')
    return parser.parse_args()

def extract_params(circuit_tokens):
    """
    Parses a Tetris circuit list of tokens into layers of (avg_beta, gamma).
    Format: ['new_layer_p', op1, b1, op2, b2, ..., gamma, 'new_layer_p', ...]
    """
    layers = []
    i = 0
    while i < len(circuit_tokens):
        if circuit_tokens[i] == 'new_layer_p':
            i += 1
            # Find next marker
            next_m = i
            while next_m < len(circuit_tokens) and circuit_tokens[next_m] != 'new_layer_p':
                next_m += 1
            
            layer_tokens = circuit_tokens[i:next_m]
            if len(layer_tokens) >= 3:
                try:
                    gamma = float(layer_tokens[-1])
                    # Beta values are at indices 1, 3, 5... of the layer_tokens list
                    # since 0, 2, 4... are op indices.
                    betas = []
                    for k in range(1, len(layer_tokens) - 1, 2):
                        betas.append(float(layer_tokens[k]))
                    
                    if betas:
                        avg_beta = np.mean(betas)
                        layers.append((avg_beta, gamma))
                except (ValueError, TypeError):
                    pass # Skip malformed layers
            i = next_m
        else:
            i += 1
    return layers

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.input, 'r') as f:
        data = json.load(f)

    # Separate by formula type: { 'balanced': [layers_per_sample], 'random': [...] }
    type_data = {'balanced': [], 'random': []}

    for entry in data:
        f_type = entry.get('type', 'unknown')
        if f_type not in type_data:
            type_data[f_type] = []
        
        # entry['q_circuits'] is a list of N samples
        for sample_circ in entry.get('q_circuits', []):
            layers = extract_params(sample_circ)
            if layers:
                type_data[f_type].append(layers)

def get_agg_stats(circuits):
    if not circuits:
        return None, None, None, None, 0
    
    max_p = max(len(c) for c in circuits)
    beta_stats = [[] for _ in range(max_p)]
    gamma_stats = [[] for _ in range(max_p)]

    for c in circuits:
        for p_idx, (b, g) in enumerate(c):
            beta_stats[p_idx].append(b)
            gamma_stats[p_idx].append(g)

    beta_means = [np.mean(b) if b else 0 for b in beta_stats]
    beta_stds  = [np.std(b) if b else 0 for b in beta_stats]
    gamma_means = [np.mean(g) if g else 0 for g in gamma_stats]
    gamma_stds  = [np.std(g) if g else 0 for g in gamma_stats]
    
    return beta_means, beta_stds, gamma_means, gamma_stds, max_p

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.input, 'r') as f:
        data = json.load(f)

    # Separate by formula type
    # Each list will store 'layers' (list of (beta, gamma))
    gpt_data = {'balanced': [], 'random': []}
    adapt_data = {'balanced': [], 'random': []}

    for entry in data:
        f_type = entry.get('type', 'unknown')
        if f_type not in gpt_data:
            gpt_data[f_type] = []
            adapt_data[f_type] = []
        
        # GPT Predictions (from samples)
        for sample_circ in entry.get('q_circuits', []):
            layers = extract_params(sample_circ)
            if layers:
                gpt_data[f_type].append(layers)
        
        # ADAPT Ground Truth
        if 'adapt_circuit' in entry:
            layers = extract_params(entry['adapt_circuit'])
            if layers:
                adapt_data[f_type].append(layers)

def extract_operator_indices(circuit_tokens):
    """Parses a Tetris circuit list and returns all operator indices found."""
    indices = []
    i = 0
    while i < len(circuit_tokens):
        if circuit_tokens[i] == 'new_layer_p':
            i += 1
            # In each layer, tokens are pairs of (op_idx, beta) followed by gamma
            # Find next marker
            next_m = i
            while next_m < len(circuit_tokens) and circuit_tokens[next_m] != 'new_layer_p':
                next_m += 1
            
            layer_tokens = circuit_tokens[i:next_m]
            if len(layer_tokens) >= 3:
                # 0, 2, 4... are op indices
                for k in range(0, len(layer_tokens) - 1, 2):
                    try:
                        indices.append(int(layer_tokens[k]))
                    except (ValueError, TypeError):
                        pass
            i = next_m
        else:
            i += 1
    return indices

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.input, 'r') as f:
        data = json.load(f)

    # Buckets for angles (reuse previous structures)
    gpt_angle_data = {'balanced': [], 'random': []}
    adapt_angle_data = {'balanced': [], 'random': []}
    
    # Buckets for operator indices
    gpt_op_indices = {'balanced': [], 'random': []}
    adapt_op_indices = {'balanced': [], 'random': []}

    for entry in data:
        f_type = entry.get('type', 'unknown')
        if f_type not in gpt_angle_data:
            gpt_angle_data[f_type] = []
            adapt_angle_data[f_type] = []
            gpt_op_indices[f_type] = []
            adapt_op_indices[f_type] = []
        
        # GPT Predictions
        for sample_circ in entry.get('q_circuits', []):
            layers = extract_params(sample_circ)
            if layers:
                gpt_angle_data[f_type].append(layers)
            gpt_op_indices[f_type].extend(extract_operator_indices(sample_circ))
        
        # ADAPT Ground Truth
        if 'adapt_circuit' in entry:
            layers = extract_params(entry['adapt_circuit'])
            if layers:
                adapt_angle_data[f_type].append(layers)
            adapt_op_indices[f_type].extend(extract_operator_indices(entry['adapt_circuit']))

    for f_type in gpt_angle_data.keys():
        # 1. Plot Parameter Comparison
        g_b_mean, g_b_std, g_g_mean, g_g_std, g_max_p = get_agg_stats(gpt_angle_data[f_type])
        a_b_mean, a_b_std, a_g_mean, a_g_std, a_max_p = get_agg_stats(adapt_angle_data[f_type])

        if g_max_p > 0 or a_max_p > 0:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            # [Previous 2x2 plotting code for angles remains same in logic...]
            if g_max_p > 0:
                idx = np.arange(1, g_max_p+1)
                axes[0,0].plot(idx, g_b_mean, 'b-o', label='Q3SAT-GPT Beta')
                axes[0,0].fill_between(idx, np.array(g_b_mean)-g_b_std, np.array(g_b_mean)+g_b_std, alpha=0.2, color='blue')
                axes[0,1].plot(idx, g_g_mean, 'r-s', label='Q3SAT-GPT Gamma')
                axes[0,1].fill_between(idx, np.array(g_g_mean)-g_g_std, np.array(g_g_mean)+g_g_std, alpha=0.2, color='red')
            
            if a_max_p > 0:
                idx = np.arange(1, a_max_p+1)
                axes[1,0].plot(idx, a_b_mean, 'c-o', label='MosaicADAPT-QAOA Beta')
                axes[1,0].fill_between(idx, np.array(a_b_mean)-a_b_std, np.array(a_b_mean)+a_b_std, alpha=0.2, color='cyan')
                axes[1,1].plot(idx, a_g_mean, 'tab:orange', marker='s', label='MosaicADAPT-QAOA Gamma')
                axes[1,1].fill_between(idx, np.array(a_g_mean)-a_g_std, np.array(a_g_mean)+a_g_std, alpha=0.2, color='orange')

            for row in range(2):
                for col in range(2):
                    ax = axes[row, col]
                    ax.set_ylim(-1.0, 1.0)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    ax.set_xlabel('Layer')
                    if col == 0:
                        ax.set_ylabel('beta value', labelpad=5)
                    else:
                        ax.set_ylabel('gamma value', labelpad=5)
            
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            plt.savefig(os.path.join(args.out_dir, f'variance_comparison_{f_type}.png'), bbox_inches='tight')
            plt.close()

        # 2. Plot Operator Frequency Comparison
        if gpt_op_indices[f_type] or adapt_op_indices[f_type]:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.hist(gpt_op_indices[f_type], bins=50, color='blue', alpha=0.7, label='Q3SAT-GPT Selected')
            plt.title(f'Q3SAT-GPT Operator Frequency ({f_type})')
            plt.xlabel('Operator Index'); plt.ylabel('Count'); plt.legend(); plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            plt.hist(adapt_op_indices[f_type], bins=50, color='cyan', alpha=0.7, label='MosaicA-QAOA Selected')
            plt.title(f'MosaicA-QAOA Operator Frequency ({f_type})')
            plt.xlabel('Operator Index'); plt.ylabel('Count'); plt.legend(); plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f'operator_frequency_{f_type}.png'))
            plt.close()

        print(f"Update: Generated variance and operator plots for {f_type}")

if __name__ == "__main__":
    main()
