#!/bin/bash
# Run QAOA on dataset

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Set these variables manually
#################################
# Set this to be the directory containing the dataset
DATASET_FINAL=""
# Set this to be the directory where the results will be stored
RESULTS_DIR=""
# Number of tasks to submit (size of the array job)
NUM_TASKS=200
# Initial guess for gamma - this is NOT used anymore as we do a grid search over 3 gammas
INITIAL_GAMMA="0.5"
# Method to use (either "greedy" or "kamis"). kamis corresponds to MosaicADAPT-QAOA.
METHOD_NAME="kamis"
LAYER_LIMIT="20"
POOL_TYPE="qaoa_nondiagonal_double_pool"
#################################

if [ ! -d "$DATASET_FINAL" ]; then
    echo "Error: Dataset directory $DATASET_FINAL does not exist."
    exit 1
fi

mkdir -p "$RESULTS_DIR/logs"

echo "=== Starting QAOA on $DATASET_FINAL ==="

# Pick all instances under $DATASET_FINAL.
echo "Mapping all instances under: $DATASET_FINAL"

# Dump all CNF file absolute paths into a    single text manifest
find "$DATASET_FINAL" -name "*.cnf" | sort > "$RESULTS_DIR/instance_list.txt"
COUNT=$(wc -l < "$RESULTS_DIR/instance_list.txt")

if [ "$COUNT" -eq 0 ]; then
    echo "No CNF files found! Aborting."
    exit 1
fi

echo "Secured $COUNT total payload instances."

# Calculate files per task based on total count to stay within QOS limits
FILES_PER_TASK=$(( (COUNT + NUM_TASKS - 1) / NUM_TASKS ))

echo "Batching $FILES_PER_TASK files per task..."
echo "Submitting a single Array of $NUM_TASKS tasks..."

JOB_ID=$(sbatch --parsable \
    --array=1-${NUM_TASKS} \
    --output="$RESULTS_DIR/logs/gen_%A_%a.out" \
    --error="$RESULTS_DIR/logs/gen_%A_%a.err" \
    "$PROJECT_ROOT/slurm_scripts/run_mosaicadapt_singlejob.sh" "$RESULTS_DIR/instance_list.txt" "$RESULTS_DIR" "$FILES_PER_TASK" "$INITIAL_GAMMA" "$METHOD_NAME" "$LAYER_LIMIT" "$POOL_TYPE")
    
echo ""
echo "  -> Master Array Job ID: $JOB_ID"
echo "=== Job submitted successfully! ==="
echo "Monitor with: squeue -u \$(whoami)"
echo "Output JSONs will be stored in: $RESULTS_DIR/results_json/"

# This job will stay in 'Pending' status until the Array job completes successfully
COLLATE_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:$JOB_ID \
    --output="$RESULTS_DIR/logs/collation_%j.out" \
    --error="$RESULTS_DIR/logs/collation_%j.err" \
    "$PROJECT_ROOT/slurm_scripts/collate_results_slurm.sh" \
    "$DATASET_FINAL/metadata.csv" \
    "$RESULTS_DIR" \
    "$RESULTS_DIR/optimized_circuits.csv")

echo "  -> Collation Job ID: $COLLATE_JOB_ID (waiting for $JOB_ID)"
echo "=== Pipeline fully submitted! ==="
