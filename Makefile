# QAOA-GPT Makefile

.PHONY: help test-feather

help:
	@echo "Available commands:"
	@echo "  make test-feather    - Run the Perturbation Analysis (FEATHER embeddings)"

test-feather:
	@echo "Running Perturbation Analysis for FEATHER embeddings..."
	python src/qaoa-gpt/tests/feather-embeddings/test_sat_feather_embeddings.py

test-evaluate-qaoa-circuits:
	@echo "Running QAOA circuit evaluation..."
	python src/qaoa-gpt/tests/reconstruct-qaoa-circuits.py
	
test-prepare-circ:
	@echo "Running QAOA circuit preparation..."
	python src/qaoa-gpt/prepare_circ.py --csv_input src/qaoa-gpt/tests/prepare_circ/optimized_circuits.csv --save_dir src/qaoa-gpt/tests/prepare_circ/results --n_nodes 12
	
test: test-feather test-evaluate-qaoa-circuits test-prepare-circ
