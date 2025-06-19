#!/bin/bash

# Create timestamp for unique log directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="experiment_starting_year_1_ablation_no_topic_model_drop_team_members_logs_${TIMESTAMP}_csp0.5_lr2e-2_wd5e-5_hd32_beta0_min1e-5_patience4_factor0.5_early8"
mkdir -p "$LOG_DIR"

# Set up logging
exec 1> >(tee -a "${LOG_DIR}/experiment_master.log")
exec 2>&1

echo "Starting experiments at $(date)"
echo "Log directory: ${LOG_DIR}"

# Ablation options for --inference_time_author_dropping
ABLATIONS=(
    "no author"
    "all authors"
    "drop_first"
    "drop_last"
    "keep_first"
    "keep_last"
    "drop_first_and_last"
)

# List of CUDA devices to use (edit as needed)
CUDA_DEVICES=(0 1 2 3 4)

# Path to checkpoint
CHECKPOINT="runs/20250619_001921_team/evaluated_model_epoch070_male0_0.682_male1_0.718_male2_0.736_male3_0.760_male4_0.767_team.pt"

# Function to run a single experiment
run_experiment() {
    local cuda_id=$1
    local run_id=$2
    local drop_fn="$3"

    echo "Starting experiment $run_id on cuda:$cuda_id with --inference_time_author_dropping '$drop_fn'"
    nohup python train.py \
        --train_years 2006 2014 \
        --test_years 2015 2018 \
        --lr 2e-2 \
        --weight_decay 5e-5 \
        --training_off 1 \
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
        --input_feature_model 'drop topic' \
        --inference_time_author_dropping "$drop_fn" \
        --load_checkpoint "$CHECKPOINT" \
        --device cuda:$cuda_id \
        > "${LOG_DIR}/experiment_${run_id}_cuda${cuda_id}_${drop_fn// /_}.log" 2>&1 &
}

# Run ablations, assigning each to a CUDA device (cycling if needed)
run_id=1
for i in "${!ABLATIONS[@]}"; do
    cuda_idx=$((i % ${#CUDA_DEVICES[@]}))
    cuda_id=${CUDA_DEVICES[$cuda_idx]}
    drop_fn="${ABLATIONS[$i]}"
    run_experiment $cuda_id $run_id "$drop_fn"
    ((run_id++))
done

# Print the process IDs for reference
echo "All experiments started. Process IDs:"
jobs -p

echo "You can now close this terminal. Check ${LOG_DIR} for logs."
echo "To check running processes later, use: ps aux | grep train.py"