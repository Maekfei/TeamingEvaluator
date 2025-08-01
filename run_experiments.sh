#!/bin/bash

# Create timestamp for unique log directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="ablation_no_author_experiment_add_social_starting_year_1_logs_${TIMESTAMP}_csp0.5_lr1e-2_wd5e-5_hd32_beta0_min1e-5_patience4_factor0.5_early8_avgref33"
mkdir -p "$LOG_DIR"

# Set up logging
exec 1> >(tee -a "${LOG_DIR}/experiment_master.log")
exec 2>&1

echo "Starting experiments at $(date)"
echo "Log directory: ${LOG_DIR}"

# Function to run a single experiment
run_experiment() {
    local cuda_id=$1
    local run_id=$2

    echo "Starting experiment $run_id on cuda:$cuda_id"
    
    nohup python train.py \
        --train_years 2006 2014 \
        --test_years 2015 2018 \
        --lr 1e-2 \
        --weight_decay 5e-5 \
        --hidden_dim 32 \
        --epochs 800 \
        --cold_start_prob 0.5 \
        --beta 0 \
        --grad_clip 1.0 \
        --lr_patience 4 \
        --lr_factor 0.5 \
        --early_stop_patience 8 \
        --min_lr 1e-5 \
        --eval_mode team \
        --input_feature_model "drop authors"\
        --device cuda:$cuda_id \
        > "${LOG_DIR}/experiment_${run_id}_cuda${cuda_id}.log" 2>&1 &
}

# Run 10 experiments with the same configuration
# cuda:5
run_experiment 5 1
run_experiment 5 2

# # cuda:6
run_experiment 6 3
run_experiment 6 4

# # cuda:7
run_experiment 7 5
run_experiment 7 6


# Print the process IDs for reference
echo "All experiments started. Process IDs:"
jobs -p

echo "You can now close this terminal. Check ${LOG_DIR} for logs."
echo "To check running processes later, use: ps aux | grep train.py"