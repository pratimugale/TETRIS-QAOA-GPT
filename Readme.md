# QAOA-GPT and TETRIS-ADAPT-QAOA for Max-E3-SAT

This repo contains code for the work:. It built on top of a fork of the original QAOA-GPT repo: [QAOA-GPT](https://github.com/IlyaTyagin/ADAPT-GPT).

## Quick Start Guide

**Installing**:
- Julia part:
	1. Install Julia: https://julialang.org/downloads/ 
	2. For your convenience, you can also add Julia kernel to your Jupyter: https://julialang.github.io/IJulia.jl/stable/manual/installation/ (Julia notebooks are not used in this repo, but can be handy for debugging)
- Python part:
	1. Create conda environment:  `conda create -n adapt_gpt python=3.10`
	2. Activate it: `conda activate adapt_gpt`
	3. Install python dependencies: `pip install torch numpy transformers datasets tiktoken wandb ipykernel pandas tqdm networkx matplotlib joblib scipy gurobipy scikit-learn`
3. ADAPT GPT codebase:
	1. Clone this repo with its dependencies: `git clone https://github.com/pratimugale/TETRIS-QAOA-GPT --recurse-submodules`
	2. `cd TETRIS-QAOA-GPT/TetrisADAPT.jl/`
	3. Run julia: `julia --project=.`
	4. Install Julia dependencies. Inside julia interpreter run: `julia> using Pkg; Pkg.instantiate(); Pkg.add(["JuMP", "MQLib" , "ProgressBars", "SimpleWeightedGraphs", "CSV", "DataFrames", "JSON", "ArgParse", "Multibreak"]); Pkg.develop(path="SciPyOptimizers");` 

**Running**:

The pipeline is run as follows:
1. Generate graph-circuit pairs with ADAPT.jl:
	- Run multithreaded ADAPT (you can edit the `adapt_maxcut_run_multithread.sh` script to adjust the parameters): `./adapt_maxcut_run_multithread.sh`
	- Note: for a meaningful GPT model, we need at least 50k circuits. This number usually requires a CPU-based cluster.
2. Tokenize them and prepare for GPT training:
	- Run: `python prepare_circ.py --adapt_results_dir <ADAPT_RESULTS_DIR> --save_dir <SAVING_DIR> --n_nodes <N_NODES>`, where `<N_NODES>` is the problem size (qubits/graph nodes) for all circuits in the dataset.
	- Note: nanoGPT expects `<SAVING_DIR>` to be a folder inside `nanoGPT/data`
3. Train GPT model:
	1. Go to nanoGPT directory: `cd nanoGPT/`
	2. Run: `python train_pad_gemb_ar_eval.py <SAVING_DIR>/train_adapt_gpt_config.py`
4. Use a pre-trained model for inference:
	1. Generate circuits for random graphs and evaluate them with ADAPT.jl. Notebook: `qaoa_gpt_inference_demo.ipynb`

## Data Availability 

Pre-trained models that we used to generate results in our paper are available here: [TODO].

## Tests
Run `make test` to run all tests for helper functions and utilities.

## Citing 

