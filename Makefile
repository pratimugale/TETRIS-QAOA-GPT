# QAOA-GPT Makefile

.PHONY: help test-feather test-evaluate-qaoa-circuits test-prepare-circ test-tetris-reproduce test

help:
	@echo "Available commands:"
	@echo "  make test-feather           - Run the Perturbation Analysis (FEATHER embeddings)"
	@echo "  make test-evaluate-qaoa      - Run the standard QAOA circuit evaluation"
	@echo "  make test-tetris-reproduce  - Run the bit-perfect Tetris-QAOA reproduction"
	@echo "  make test                   - Run all tests"

test-feather:
	@echo "Running Perturbation Analysis for FEATHER embeddings..."
	python src/qaoa-gpt/tests/feather-embeddings/test_sat_feather_embeddings.py

test-reconstruct-qaoa-circuits:
	@echo "Running standard QAOA circuit evaluation..."
	python src/qaoa-gpt/tests/reconstruct-qaoa-circuits.py

test-prepare-circ:
	@echo "Running QAOA circuit preparation..."
	python src/qaoa-gpt/prepare_circ.py --csv_input src/qaoa-gpt/tests/prepare_circ/optimized_circuits.csv --save_dir src/qaoa-gpt/tests/prepare_circ/results --n_nodes 12

test-reconstruct-tetris-circuits:
	@echo "Running bit-perfect Tetris-QAOA energy reproduction..."
	python src/qaoa-gpt/tests/reconstruct-tetris-circuits.py

test: test-feather test-reconstruct-qaoa-circuits test-prepare-circ test-reconstruct-tetris-circuits
