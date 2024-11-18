#!/bin/bash

export OPENBLAS_NUM_THREADS=1
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

#######################################
# ADAPT parameters (adjust as needed) #
N_WORKERS=20
OUTPUT_DIR="<YOUR_DIRECTORY_HERE>"
GRAPHS_NUMBER=100
N_NODES=12
TRIALS_PER_GRAPH=3
ENERGY_TOL_FRAC="0.03"
MAX_PARAMS=100
WEIGHTED=false
NORMALIZE_WEIGHTS=true
#######################################

CUR_DATE=$(date +'%Y-%m-%d_%H-%M')
LOG_DIR=$OUTPUT_DIR/logs/$CUR_DATE
mkdir -p $LOG_DIR

###########################
# ADAPT Parallel executor # 
for ((i=0; i<N_WORKERS; i++))
do
    echo "Starting copy" $i " on node: " $HOSTNAME
    julia --project=$SCRIPT_DIR/ADAPT.jl $SCRIPT_DIR/adapt_maxcut_run_1thread.jl \
        --output-dir $OUTPUT_DIR \
        --graphs-number $GRAPHS_NUMBER \
        --n-nodes $N_NODES \
        --trials-per-graph $TRIALS_PER_GRAPH \
        --energy-tol-frac $ENERGY_TOL_FRAC \
        --max-params $MAX_PARAMS \
        --weighted $WEIGHTED \
        --run-qaoa true \
        --run-vqe true \
        --calc-h-eigen true \
        --run-diag-qaoa true \
        --normalize-weights $NORMALIZE_WEIGHTS \
        2>&1 | tee $LOG_DIR/worker_${HOSTNAME}_${i}.log \
        & 
done
wait
