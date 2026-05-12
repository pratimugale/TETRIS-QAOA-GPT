#!/bin/bash
#SBATCH --job-name=mosaicadapt_20layer_nddp_n12
#SBATCH --partition=idle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=12:00:00

PROJECT_ROOT="$SLURM_SUBMIT_DIR"

INSTANCE_LIST=$1
MASTER_RUN_DIR=$2
FILES_PER_TASK=${3:-600}

if [ -z "$INSTANCE_LIST" ] || [ -z "$MASTER_RUN_DIR" ]; then
    echo "Usage: sbatch slurm_scripts/run_qaoa_slurm.sh <instance_list.txt> <master_run_dir> <files_per_task> <gamma> <method> <layers> <pool>"
    exit 1
fi

INITIAL_GAMMA=${4:-0.5} # TODO: This is not used anymore as we do a grid search- remove later
METHOD_NAME=${5:-kamis}
LAYER_LIMIT=${6:-5}
POOL_TYPE=${7:-qaoa_mixer}

# Explicitly isolate threads for high-throughput single-core array efficiency
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Calculate precisely which subset of files this single Slurm Array task is exclusively responsible for
OFFSET=$(( (${SLURM_ARRAY_TASK_ID} - 1) * FILES_PER_TASK + 1 ))
END=$(( OFFSET + FILES_PER_TASK - 1 ))

echo "Task ${SLURM_ARRAY_TASK_ID} processing lines $OFFSET to $END"
echo "Evaluator booted on $(hostname) at $(date)"

# Distribute 12 jobs per task internally - to parallelize at a process level
CONCURRENCY=12
ACTIVE_PROCS=0

# Loop natively over the exact contiguous line chunk allocated
for IDX in $(seq $OFFSET $END); do
    MAP_FILE=$(sed -n "${IDX}p" "$INSTANCE_LIST")
    
    if [ -z "$MAP_FILE" ]; then
        continue
    fi
    
    # Sub-divide massive JSON output into 100 clean shard directories
    SUBDIR=$(printf "%02d" $((${IDX} % 100)))
    OUT_DIR="$MASTER_RUN_DIR/results_json/$SUBDIR"
    mkdir -p "$OUT_DIR"

    LABEL=$(basename "$MAP_FILE" .cnf)
    
    # Since start_qaoa.jl now runs a grid search over 3 gammas [0.01, 0.1, 0.5]
    # We check if ALL THREE exist. If they do, we skip the processing for that instance.
    G01_JSON="$OUT_DIR/qaoa_tetris_adapt_qaoa_${METHOD_NAME}_gamma0.01_${LABEL}.json"
    G1_JSON="$OUT_DIR/qaoa_tetris_adapt_qaoa_${METHOD_NAME}_gamma0.1_${LABEL}.json"
    G5_JSON="$OUT_DIR/qaoa_tetris_adapt_qaoa_${METHOD_NAME}_gamma0.5_${LABEL}.json"

    if [ -s "$G01_JSON" ] && [ -s "$G1_JSON" ] && [ -s "$G5_JSON" ]; then
        echo "Skip: Full grid search (0.01-0.5) already exists for $LABEL."
        continue
    fi

    echo "Running: $MAP_FILE (Layer limit: $LAYER_LIMIT, Pool: $POOL_TYPE)"
    
    # Launch with the 5th argument for layer limit and 6th for pool type
    julia --project="$PROJECT_ROOT" "$PROJECT_ROOT/src/qaoa/start_qaoa.jl" "$MAP_FILE" "$OUT_DIR" "$INITIAL_GAMMA" "$METHOD_NAME" "$LAYER_LIMIT" "$POOL_TYPE" &
    
    ACTIVE_PROCS=$((ACTIVE_PROCS + 1))
    
    # Wait if we reached concurrency limit
    if [ "$ACTIVE_PROCS" -ge "$CONCURRENCY" ]; then
        wait
        ACTIVE_PROCS=0
    fi

done

# Wait for the remainder
wait

echo "Worker finished at $(date)"
