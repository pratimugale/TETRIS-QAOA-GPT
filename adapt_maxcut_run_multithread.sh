#!/bin/bash

export OPENBLAS_NUM_THREADS=1
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

#######################################
# ADAPT parameters (adjust as needed) #
N_WORKERS=1
OUTPUT_DIR="<YOUR_DIRECTORY_HERE>"
GRAPHS_NUMBER=10
N_NODES=8
TRIALS_PER_GRAPH=3
MAX_PARAMS=100
WEIGHTED=false
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
        --max-params $MAX_PARAMS \
        --weighted $WEIGHTED \
        --run-qaoa true \
        --run-vqe true \
        --calc_h_eigen true \
        2>&1 | tee $LOG_DIR/worker_${HOSTNAME}_${i}.log \
        & 
done
wait
