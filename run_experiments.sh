#!/bin/bash

# Set up logging
exec 1> >(tee -a "experiment_master.log")
exec 2>&1

echo "Starting experiments at $(date)"

# Function to run a single experiment
run_experiment() {
    local cuda_id=$1
    local hidden_dim=$2
    local cold_start_prob=$3
    local beta=$4
    local run_id=$5

    echo "Starting experiment $run_id on cuda:$cuda_id with hidden_dim=$hidden_dim, cold_start_prob=$cold_start_prob, beta=$beta"
    
    python train.py \
        --train_years 2006 2014 \
        --test_years 2015 2018 \
        --lr 2e-3 \
        --weight_decay 5e-4 \
        --hidden_dim $hidden_dim \
        --epochs 800 \
        --cold_start_prob $cold_start_prob \
        --beta $beta \
        --eval_mode team \
        --device cuda:$cuda_id \
        > "experiment_${run_id}_cuda${cuda_id}_hd${hidden_dim}_csp${cold_start_prob}_beta${beta}.log" 2>&1
}

# Create a directory for experiment logs
mkdir -p experiment_logs

# First batch of experiments (10 parallel runs)
# cuda:0
run_experiment 0 32 0 0 1 &
run_experiment 0 32 0 0 2 &

# cuda:1
run_experiment 1 32 0 0 3 &
run_experiment 1 32 0 0 4 &

# cuda:2
run_experiment 2 32 0 0 5 &
run_experiment 2 32 0 0 6 &

# cuda:3
run_experiment 3 32 0 0 7 &
run_experiment 3 32 0 0 8 &

# cuda:4
run_experiment 4 32 0 0 9 &
run_experiment 4 32 0 0 10 &