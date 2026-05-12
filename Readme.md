# Q3SAT-GPT

This repo contains code for Q3SAT-GPT. It built on top of a fork of the original QAOA-GPT repo: [QAOA-GPT](https://github.com/IlyaTyagin/ADAPT-GPT). Please note that some code is still being added - from discrete sources and put all together in this repo. It will be added as soon as possible. Thank you.

## Quick Start Guide

**Installing**:
- Julia part:
	1. Install Julia: https://julialang.org/downloads/ 
- Python part:
	1. Create conda environment:  `conda create -n adapt_gpt python=3.10`
	2. Activate it: `conda activate adapt_gpt`
	3. Install python dependencies: `pip install torch numpy transformers datasets tiktoken wandb ipykernel pandas tqdm networkx matplotlib joblib scipy gurobipy scikit-learn`
3. [MosaicADAPT-QAOA](https://github.com/pratimugale/TetrisADAPT.jl):
	1. Clone this repo with its dependencies: `git clone https://github.com/pratimugale/TETRIS-QAOA-GPT --recurse-submodules`
	2. `cd TETRIS-QAOA-GPT/TetrisADAPT.jl/`
	3. Run julia: `julia --project=.`
	4. Install Julia dependencies. Inside julia interpreter run: `julia> using Pkg; Pkg.instantiate(); Pkg.add(["JuMP", "MQLib" , "ProgressBars", "SimpleWeightedGraphs", "CSV", "DataFrames", "JSON", "ArgParse", "Multibreak"]); Pkg.develop(path="SciPyOptimizers");` 
	5. Install the Python workspace as an editable standard package (prevents import errors): run `cd ..` and `pip install -e .`

**Running**:

The pipeline is run as follows:
1. Generate the dataset using `make generate-dataset N_VARS=<N_VARS> N_INSTANCES=<N_INSTANCES> SEED=<SEED>`
2. Run MosaicADAPT-QAOA on all instances. Use the [scripts/run_mosaicadapt_allinstances.sh](scripts/run_mosaicadapt_allinstances.sh) script to do this on an HPC cluster (using slurm and parallize the processing at a process level). Once all instances are processed, the [scripts/collate_results_slurm.sh](scripts/collate_results_slurm.sh) script will generate a CSV file containing the results. This CSV file can be used in the next step. Note that while the code has currently been uploaded in the repo, some paths in the Julia script need updation and will be updated soon.
3. Tokenize them and prepare for GPT training:
	- Run: `python prepare_circ.py --csv_input <CSV_INPUT> --save_dir <SAVING_DIR> --n_nodes <N_NODES> --pool_type <POOL_TYPE>`, where `<N_NODES>` is the problem size (qubits/graph nodes) for all circuits in the dataset.
	- Note: nanoGPT expects `<SAVING_DIR>` to be a folder inside `nanoGPT/data`
4. Train GPT model:
	1. Go to nanoGPT directory: `cd nanoGPT/`
	2. Run: `python train_pad_gemb_ar_eval.py <SAVING_DIR>/train_adapt_gpt_config.py`
	3. **Visualize parameter distributions**:
	   ```bash
	   # General distribution plot
	   python3 src/qaoa-gpt/analysis/plot_parameter_distributions.py --input path/to/eval_output.json --out_dir analysis_plots

	   # Comparison plot (Q3SAT-GPT vs MosaicA-QAOA)
	   python3 src/qaoa-gpt/analysis/plot_gpt_vs_adapt_dist.py --input path/to/eval_output.json --out_dir analysis_plots/gpt_vs_adapt
	   ```


## Data Availability 

Pre-trained models that we used to generate results in our paper are available here: [TODO].

## Tests
Run `make test` to run all tests for helper functions and utilities.

## Citing 

