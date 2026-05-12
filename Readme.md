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
	- Example for a 12-variable dataset: 
	```sh
	python prepare_circ.py \
    --csv_input scripts/collated_results_n12.csv \
    --save_dir n12_mosaic_dataset \
    --n_nodes 12 \
    --pool_type qaoa_nondiagonal_double_pool
	```

4. Train GPT model:
	1. Go to nanoGPT directory: `cd nanoGPT/`
	2. Run: `python train_pad_gemb_ar_eval.py <SAVING_DIR>/train_adapt_gpt_config.py`. 
	Example:
	```sh
	python train_pad_gemb_ar_eval.py data/n12_mosaic_dataset/train_adapt_gpt_config.py
	```
	3. Select the checkpoint model from which the evaluation on the test set needs to be run. This can be based on the performance on the validation set. Command: 

	```sh
	python src/qaoa_gpt/eval/eval_testset_batch.py \
    --checkpoint "nanoGPT/out-q3sat-gpt-n12/<ckpt>.pt" \
    --data_dir "src/qaoa_gpt/dataset/q3sat-gpt-n12-gridsearch" \
    --n_samples 5 \
    --device cuda \
    --n_nodes 12 \
    --pool_type "qaoa_nondiagonal_double_pool" \
    --batch_size <batch size> \
    --julia_threads 4 \
    --julia_script "src/qaoa_gpt/adapt_gpt_eval_energy.jl" \
    --out_dir "results/eval_n12"

	```
	The performance on the test set in terms of metrics will be reported in the stdout of the previous script. 

5. **Visualize parameter distributions**
The parameter distribution plots shown in the paper can be visualized by running the following command:
```sh
python3 src/qaoa_gpt/analysis/plot_circuit_variance.py \
    --input results/eval_n12/testset_eval_output.json \
    --out_dir results/eval_n12/analysis_plots
```


## Data Availability 

Pre-trained models that we used to generate results in our paper are available here: [TODO].

## Tests
Run `make test` to run all tests for helper functions and utilities.

## Citing 

If you found our work useful, please cite our [arXiv preprint](https://arxiv.org/abs/2604.27324):
```
@misc{ugale2026q3satgptgenerativemodeldiscovering,
      title={Q3SAT-GPT: A Generative Model for Discovering Quantum Circuits for the 3-SAT Problem}, 
      author={Pratim Ugale and Ilya Tyagin and Karunya Shirali and Kien X. Nguyen and Ilya Safro},
      year={2026},
      eprint={2604.27324},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2604.27324}, 
}
```


