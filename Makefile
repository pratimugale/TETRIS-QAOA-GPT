# QAOA-GPT Makefile

.PHONY: help test-feather test-prepare-circ test generate-dataset

help:
	@echo "Available commands:"
	@echo "  make test-feather           - Run the Perturbation Analysis (FEATHER embeddings)"
	@echo "  make test-prepare-circ      - Run the QAOA circuit preparation"
	@echo "  make test                   - Run all tests"
	@echo "  make generate-dataset N_VARS=10 N_INSTANCES=1000 SEED=2000  - Generate 3-SAT dataset"

test-feather:
	@echo "Running Perturbation Analysis for FEATHER embeddings..."
	python src/qaoa_gpt/tests/feather-embeddings/test_sat_feather_embeddings.py

test-reconstruct-qaoa-circuits:
	@echo "Running standard QAOA circuit evaluation..."
	python src/qaoa_gpt/tests/reconstruct-qaoa-circuits.py

test-prepare-circ:
	@echo "Running QAOA circuit preparation..."
	python src/qaoa_gpt/prepare_circ.py --csv_input src/qaoa_gpt/tests/prepare_circ/optimized_circuits.csv --save_dir src/qaoa_gpt/tests/prepare_circ/results --n_nodes 12

test-reconstruct-tetris-circuits:
	@echo "Running bit-perfect Tetris-QAOA energy reproduction..."
	python src/qaoa_gpt/tests/reconstruct-tetris-circuits.py

# Default parameters for dataset generation
N_VARS ?= 10
N_INSTANCES ?= 1000
SEED ?= 2000
DATASET_DIR ?= dataset	

generate-dataset:
	@echo "Generating 3-SAT dataset (n_vars=$(N_VARS), instances=$(N_INSTANCES), seed=$(SEED))..."
	# We don't pass --split here and will handle stratified split in qaoa-gpt code
	./venv/bin/python3 -m src.qaoa_gpt.dataset_generator.run_dataset_generator --n_vars $(N_VARS) --total $(N_INSTANCES) --seed $(SEED)

test: test-feather test-reconstruct-qaoa-circuits test-prepare-circ test-reconstruct-tetris-circuits
