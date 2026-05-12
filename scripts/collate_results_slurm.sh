#!/bin/bash
#SBATCH --job-name=collate
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00

PROJECT_ROOT="$SLURM_SUBMIT_DIR"

METADATA_CSV=$1
RESULTS_DIR=$2
OUTPUT_CSV=$3

if [ -z "$METADATA_CSV" ] || [ -z "$RESULTS_DIR" ] || [ -z "$OUTPUT_CSV" ]; then
    echo "Usage: sbatch collate_results_slurm.sh <metadata.csv> <results_dir> <output.csv>"
    exit 1
fi

# Load your system's Python and GCC modules here

# Activate local virtual environment
source "$PROJECT_ROOT/venv/bin/activate"

echo "=== Starting Results Collation ==="
echo "Metadata: $METADATA_CSV"
echo "Results DIR: $RESULTS_DIR"
echo "Output: $OUTPUT_CSV"

# Run collation using the direct python environment
python3 "$PROJECT_ROOT/src/dataset/collate_results.py" \
    --metadata "$METADATA_CSV" \
    --results "$RESULTS_DIR" \
    --output "$OUTPUT_CSV"

echo "=== Collation Complete! ==="
