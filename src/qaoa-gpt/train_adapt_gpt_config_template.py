# Train an ADAPT-GPT model
# Based on https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py

out_dir = '{out_dir}'
eval_interval = 10 # Number of iterations before validation
eval_iters = 2 # Number of batches to be evaluated from the validation set
log_interval = 1 # don't print too too often
eval_ar_every = 50 # how often we do heavy approx ratio evaluation (calling ADAPT.jl)
num_samples = 5 # Number of samples to draw during training validation
eval_ar_n_samples = 50 # Number of random stratified circuits evaluated in Julia per validation

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

dataset = '{dataset}'
data_dir = 'src/qaoa-gpt/dataset/{dataset}'
gradient_accumulation_steps = 5
batch_size = 512
block_size = {block_size} # currently 512 in prepare_circ.py

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-4 # with baby networks can afford to go a bit higher
max_iters = 30000
lr_decay_iters = 30000 # make equal to max_iters usually
min_lr = 1e-5 # learning_rate / 10 usually
beta2 = 0.95 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

graph_emb_dim = 500 # default for FEATHER graph
use_graph_emb = {use_graph_emb}
eval_ar_locally = False # If True, save circuits to JSON and skip Julia evaluation
pool_type = '{pool_type}'
n_nodes = {n_nodes}
rounding_digits = 2

mask_formula_loss = True
wandb_log = True
wandb_project = 'qaoa-gpt-sat'
wandb_run_name = 'baseline'
