from pathlib import Path
import subprocess
import sys
from datetime import datetime
import pandas as pd
import torch
from tqdm import tqdm
from collections import defaultdict
import networkx as nx

from gurobipy import Model, GRB
import gurobipy as gb


def extract_graph(token_seq):
    graph_seq = []

    for idx, tok in enumerate(token_seq):
        graph_seq.append(tok)
        if tok == 'end_of_graph':
            break
    adapt_seq = token_seq[idx+1:-1]
    return graph_seq, adapt_seq

def circ_sanity_check(cur_q_circ):
    
    lr_sep_list = cur_q_circ[0::4]
    op_idx_list = cur_q_circ[1::4]

    num_vals = cur_q_circ[2::4] + cur_q_circ[3::4]

    if any(
        [type(el) != int for el in op_idx_list]
    ):
        #print('wrong op_idx_list')
        return False

    if any(
        [type(el) != str for el in lr_sep_list]
    ):
        #print('wrong lr_sep_list')
        return False
    
    if len(cur_q_circ) % 4:
        #print('Wrong length')
        return False

    return True


def generate_circ_from_df(
    test_run_df,
    graph_emb_np, # for models with graph emb
    emb_graph_id_to_idx_dict, # for models with graph emb
    meta,
    model,
    device,
    ctx,
    n_samples_per_batch,
    num_samples = 5, # number of samples to draw
    max_new_tokens = 200, # number of tokens generated in each sample
    temperature = 0.1, # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200, # retain only the top_k most likely tokens, clamp others to have 0 probability
    token_seq_col = 'token_seq_round_d2',
    normalize_weights_flag = False,
    
):
    # Batched inference based on number of edges. 
    # We group graphs with the same number of edges together
    # such that we can merge them into a tensor to keep the input length size consistent.

    if graph_emb_np is not None and emb_graph_id_to_idx_dict is not None:
        gemb_flag = True
    else:
        gemb_flag = False
    
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: [itos[i] for i in l]
    
    n_edges_to_count_dict = test_run_df['edgelist_list_len'].value_counts().to_dict()
    
    adapt_gpt_out_list_dict = defaultdict(list)
    x_list_dict = defaultdict(list)
    graph_emb_dict = defaultdict(list)
    y_dict = dict()
    
    pbar = tqdm(n_edges_to_count_dict.items())
    
    for n_edges, n_graphs in pbar:
        pbar.set_description(f"Inference. Current batch: n_edges: {n_edges}, n_graphs: {n_graphs}")
        cur_test_run_df = test_run_df[
            test_run_df['edgelist_list_len'] == n_edges
        ]
        
        for row_idx, graph_df_row in cur_test_run_df.iterrows():
        #graph_df_row = test_df.loc[graph_idx]
            start, adapt_seq = extract_graph(graph_df_row[token_seq_col])
            start_ids = encode(start)
            x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
            x_list_dict[n_edges].append(x)

            if gemb_flag:
                cur_graph_idx = emb_graph_id_to_idx_dict[graph_df_row['graph_id']]
                graph_emb_dict[n_edges].append(
                    torch.tensor(graph_emb_np[cur_graph_idx], dtype=torch.bfloat16, device=device)
                )
    
            adapt_gpt_out_dict = dict()
            adapt_gpt_out_dict['graph'] = start[1:-1]
            adapt_gpt_out_dict['n_edges'] = graph_df_row['edgelist_list_len']
            adapt_gpt_out_dict['q_circuits'] = []
            adapt_gpt_out_dict['adapt_circuit'] = adapt_seq
            adapt_gpt_out_dict['adapt_full_ar'] = graph_df_row['approx_ratio']
            adapt_gpt_out_dict['graph_prefix'] = graph_df_row['graph_id']
            adapt_gpt_out_dict['energy_mqlib'] = graph_df_row['energy_mqlib']
            adapt_gpt_out_dict['label'] = graph_df_row['label']
            adapt_gpt_out_list_dict[n_edges].append(adapt_gpt_out_dict)
        
        cur_batch_torch = torch.vstack(x_list_dict[n_edges])
        
        if gemb_flag:
            cur_emb_batch_torch = torch.vstack(graph_emb_dict[n_edges])
    
        # Calculate total samples and number of mini-batches
        total_samples = cur_batch_torch.size(0)
        n_batches = (total_samples + n_samples_per_batch - 1) // n_samples_per_batch  # Ensure ceiling division
    
        # Initialize an empty list for results
        y_list = []
        
        # Run inference in mini-batches
        with torch.no_grad():
            for i in tqdm(range(n_batches), desc='Internal batch progress', disable=True):
                start_idx = i * n_samples_per_batch
                end_idx = min((i + 1) * n_samples_per_batch, total_samples)
                
                mini_batch = cur_batch_torch[start_idx:end_idx]
                mini_batch_repeated = mini_batch.repeat(num_samples, 1) # Repeat the mini-batch for num_samples

                if gemb_flag:
                    mini_emb_batch = cur_emb_batch_torch[start_idx:end_idx]
                    mini_emb_batch_repeated = mini_emb_batch.repeat(num_samples, 1) # Repeat the mini-batch for num_samples
        
                with ctx:
                    if gemb_flag:
                        y = model.generate(
                            mini_batch_repeated,
                            mini_emb_batch_repeated,
                            max_new_tokens,
                            temperature=temperature,
                            top_k=top_k
                        )
                    else:
                        y = model.generate(
                            mini_batch_repeated,
                            #mini_emb_batch_repeated,
                            max_new_tokens,
                            temperature=temperature,
                            top_k=top_k
                        )
        
                # Collect results from each mini-batch
                y_list.append(y.detach().cpu())
        
        # Concatenate results from all mini-batches
        y_dict[n_edges] = torch.cat(y_list, dim=0)

    
    ### trimming the records (removing garbage after EOS)
    for n_edges, cur_adapt_gpt_out_list in adapt_gpt_out_list_dict.items():
        cur_full_y_tensor = y_dict[n_edges]
        
        for graph_idx in range(len(cur_adapt_gpt_out_list)):
            
            cur_y_tensor = cur_full_y_tensor[graph_idx::len(cur_adapt_gpt_out_list)]
            
            for k in range(num_samples):
                cur_gen_result = decode(cur_y_tensor[k].tolist())
                cur_circ = []
                circ_flag = 0
                for idx, tok in enumerate(cur_gen_result):
                    if tok == 'end_of_graph':
                        circ_flag = 1
                    if circ_flag:
                        cur_circ.append(tok)
                    if tok == 'eos':
                        break
                cur_adapt_gpt_out_list[graph_idx]['q_circuits'].append(cur_circ[1:-1])

        ### flattening the circ list
        adapt_gpt_test_samples_list = []
        for n_edges, cur_adapt_gpt_out_list in adapt_gpt_out_list_dict.items():
            adapt_gpt_test_samples_list += cur_adapt_gpt_out_list

    for idx in range(len(adapt_gpt_test_samples_list)):
        q_circ_filt_list = []
        for circ in adapt_gpt_test_samples_list[idx]['q_circuits']:
            filt_flag = circ_sanity_check(circ)
            # if not filt_flag:
            #     #print(adapt_gpt_test_samples_list[idx]['graph_prefix'], '\n')
            #     pass
            # else:
            #     q_circ_filt_list.append(circ)
            q_circ_filt_list.append(circ)

    adapt_gpt_test_samples_list[idx]['q_circuits'] = q_circ_filt_list

    for gr_dict in adapt_gpt_test_samples_list:
        graph_jl_list = []
    
        graph_edges_list = gr_dict['graph'][::2]
        graph_weights_list = gr_dict['graph'][1::2]
    
        if normalize_weights_flag:
            graph_w_norm = sum(graph_weights_list)
        else:
            graph_w_norm = 1.0
        
        for edge_idx, edge in enumerate(graph_edges_list):
            cur_edge = list(edge)
            cur_edge += [graph_weights_list[edge_idx]/graph_w_norm]
            graph_jl_list.append(cur_edge)
    
        gr_dict['graph_w_jl'] = graph_jl_list
        gr_dict['graph_weight_norm'] = graph_w_norm

    ## make it more error-prone

    adapt_gpt_test_samples_filt_list = []
    
    for rec in adapt_gpt_test_samples_list:
        pos_flag = 1
        # if len(rec['adapt_circuit']) % 4:
        #     pos_flag = 0
        # for gpt_circ in rec['q_circuits']:
        #     if len(gpt_circ) % 4:
        #         pos_flag = 0
        
        if pos_flag:
            adapt_gpt_test_samples_filt_list.append(rec)

    adapt_gpt_test_samples_df = pd.DataFrame(adapt_gpt_test_samples_filt_list)

    return adapt_gpt_test_samples_df


def eval_adapt_gpt_circ_jl(
    adapt_gpt_res_df,
    adapt_gpt_path,
    temp_folder,
    n_nodes,
    n_threads=4,
    pool_type="qaoa_double_pool",
):
    formatted_timestamp = datetime.now().strftime('%Y-%m-%d__%H_%M_%S')

    adapt_gpt_path = Path(adapt_gpt_path)
    temp_folder = Path(temp_folder)

    temp_folder.mkdir(parents=True, exist_ok=True)

    prefix = f'adapt_gpt_res_{formatted_timestamp}_df'
    in_fname = f'{prefix}.json'
    out_fname = f'{prefix}_jl.json'

    in_fname_path = temp_folder.joinpath(in_fname).resolve()
    out_fname_path = temp_folder.joinpath(out_fname).resolve()
    
    adapt_gpt_res_df.to_json(
        in_fname_path,
        orient='records'
    )

    adapt_jl_path = adapt_gpt_path.joinpath('ADAPT.jl').resolve()
    script_path = adapt_gpt_path.joinpath('adapt_gpt_eval_energy.jl').resolve()
    process = subprocess.Popen(
        [
            "julia",
            f"-t {n_threads}",
            f"--project={adapt_jl_path}",
            script_path,
            in_fname_path,
            out_fname_path,
            str(n_nodes),
            pool_type,
        ],
        stdout=sys.stdout, 
        stderr=sys.stderr,
        text=True
    )
    
    # Wait for Julia to finish
    process.wait()

    adapt_gpt_res_w_energies_df = pd.read_json(out_fname_path)
    
    return adapt_gpt_res_w_energies_df


def elist_to_nx(input_elist, jl_idx_shift=True):
    elist = []
    if jl_idx_shift:
        for u,v,w in input_elist:
            elist.append((u-1,v-1,w))
    else:
        elist = input_elist
    
    G = nx.Graph()
    G.add_weighted_edges_from(elist)
    
    return G

def gurobi_max_cut_val(elist):
    graph = elist_to_nx(elist)
    model = Model("Max-Cut")
    model.setParam('OutputFlag', False) 
    model.setParam(GRB.Param.TimeLimit, 10)
    variables = {}
    for node in graph.nodes:
        variables[node] = model.addVar(vtype=GRB.BINARY, name=f"x_{node}")

    objective = 0
    for u,v,w in graph.edges(data="weight"):
        objective -= w*((2*variables[v]*variables[u]) - (variables[v] + variables[u]))

    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()
    solution = [variables[node].x for node in graph.nodes]
    
    return -model.ObjVal