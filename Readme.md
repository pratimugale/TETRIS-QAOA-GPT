Quick start guide, work in progress, stay tuned!

**Installing**:
- Julia part:
	1. Install Julia: https://julialang.org/downloads/ 
	2. For your convenience, you can also add Julia kernel to your Jupyter: https://julialang.github.io/IJulia.jl/stable/manual/installation/ (Julia notebooks are not used in this repo, but can be handy for debugging)
- Python part:
	1. Create conda environment:  `conda create -n adapt_gpt python=3.10`
	2. Activate it: `conda activate adapt_gpt`
	3. Install python dependencies: `pip install torch numpy transformers datasets tiktoken wandb ipykernel pandas tqdm networkx matplotlib`
3. ADAPT GPT codebase:
	1. Clone this repo with its dependencies: `git clone https://github.com/IlyaTyagin/ADAPT-GPT --recurse-submodules`
	2. `cd ADAPT-GPT/ADAPT.jl/`
	3. Run julia: `julia --project=.`
	4. Install Julia dependencies. Inside julia interpreter run: `julia> using Pkg; Pkg.instantiate(); Pkg.add(["JuMP", "MQLib" , "ProgressBars", "SimpleWeightedGraphs", "CSV", "DataFrames", "JSON", "ArgParse"]);`

**Running**:

The pipeline is run as follows:
1. Generate graph-circuit pairs with ADAPT.jl
	- Run multithreaded ADAPT (you can edit the `adapt_maxcut_run_multithread.sh` script to adjust the parameters): `./adapt_maxcut_run_multithread.sh`
	- Note: for a meaningful GPT model, we need at least 50k circuits. This number usually requires a CPU-based cluster.
2. Tokenize them and prepare for GPT training:
	- Run: `python prepare_circ.py --adapt_results_dir <ADAPT_RESULTS_DIR> --save_dir <SAVING_DIR>`
	- Note: nanoGPT expects `<SAVING_DIR>` to be a folder inside `nanoGPT/data`
3. Train GPT model
	1. Go to nanoGPT directory: `cd nanoGPT/`
	2. Run: `python train.py <SAVING_DIR>/train_adapt_gpt_config.py`
4. Evaluate the results
	1. Generate circuits with the trained model. Notebook: `adapt_gpt_gen_py.ipynb`
	2. Get their energy estimations with ADAPT. For that run: `julia adapt_gpt_eval_energy.jl <input_fpath> <output_fpath> <n_nodes>`
	3. Visualize results. Notebook: `adapt_gpt_vis_results_py.ipynb` 