import pickle
from contextlib import nullcontext
import torch
import tiktoken
from nanoGPT.model_pad_gemb import GPTConfig as GPTConfig_gemb
from nanoGPT.model_pad_gemb import GPT as GPT_gemb

from nanoGPT.model_pad import GPTConfig as GPTConfig_nogemb
from nanoGPT.model_pad import GPT as GPT_nogemb

import pandas as pd
import json
from tqdm import tqdm
import random
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
from pathlib import Path

from util import generate_circ_from_df, eval_adapt_gpt_circ_jl

class AdaptGPT():
    def __init__(
        self,
        out_dir,
        model_name,
        data_dir,
        use_graph_emb,
        n_nodes,
        temp_folder='adapt_gpt_temp_data'
    ):

        self.out_dir = Path(out_dir)
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.use_graph_emb = use_graph_emb
        self.temp_folder = Path(temp_folder)
        
        self.seed = 1337
        self.init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
        self.device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
        self.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
        self.compile = False # use PyTorch 2.0 to compile the model to be faster
        #exec(open(nanogpt_path.joinpath('configurator.py')).read()) # overrides from command line or config file
        # -----------------------------------------------------------------------------
        
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu' # for later use in torch.autocast
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)

        self.meta = pd.read_pickle(f'{data_dir}/meta.pkl')

        if use_graph_emb:
            self.gptconfig = GPTConfig_gemb
            self.gpt = GPT_gemb
        else:
            self.gptconfig = GPTConfig_nogemb
            self.gpt = GPT_nogemb

        self.model = self.open_model(
            self.out_dir,
            self.model_name
        )
            
        return None

    def open_model(
        self,
        out_dir,
        model_name,
    ):
        # init from a model saved in a specific directory
        out_path = Path(out_dir)
    
        ckpt_path = out_path / model_name
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        gptconf = self.gptconfig(**checkpoint['model_args'])
        model = self.gpt(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
    
        print()
    
        return model

    def extract_graph(self, token_seq):
        graph_seq = []
    
        for idx, tok in enumerate(token_seq):
            graph_seq.append(tok)
            if tok == 'end_of_graph':
                break
        adapt_seq = token_seq[idx+1:-1]
        return graph_seq, adapt_seq 
        
        
        