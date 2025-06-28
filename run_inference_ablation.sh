#!/bin/bash

# Simple Inference-time Ablation Experiments Script

# Configuration
BASE_CHECKPOINT="runs/20250626_193807_team/evaluated_model_epoch030_male0_0.673_male1_0.696_male2_0.709_male3_0.733_male4_0.729_team.pt"
LOG_DIR="logs/inference_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Function to run a single ablation experiment
run_experiment() {
    local cuda_id=$1
    local experiment_name=$2
    local author_dropping=$3
    local topic_dropping=$4
    
    echo "Starting: $experiment_name on cuda:$cuda_id"
    
    # Create descriptive log filename
    local log_file="${LOG_DIR}/${experiment_name}_cuda${cuda_id}.log"
    
    # Build the command
    local cmd="python train.py"
    cmd="$cmd --train_years 2006 2014"
    cmd="$cmd --training_off 1"
    cmd="$cmd --test_years 2015 2018"
    cmd="$cmd --hidden_dim 32"
    cmd="$cmd --eval_mode team"
    cmd="$cmd --device cuda:$cuda_id"
    cmd="$cmd --load_checkpoint $BASE_CHECKPOINT"
    
    # Add ablation parameters - handle spaces properly
    if [ "$author_dropping" != "none" ]; then
        if [ "$author_dropping" = "no author" ]; then
            cmd="$cmd --inference_time_author_dropping 'no author'"
        else
            cmd="$cmd --inference_time_author_dropping $author_dropping"
        fi
    fi
    
    if [ "$topic_dropping" != "none" ]; then
        if [ "$topic_dropping" = "drop topic" ]; then
            cmd="$cmd --inference_time_topic_dropping 'drop topic'"
        else
            cmd="$cmd --inference_time_topic_dropping $topic_dropping"
        fi
    fi
    
    # Run the experiment and save output to file
    echo "Running: $cmd"
    echo "Log file: $log_file"
    eval "nohup $cmd > $log_file 2>&1 &"
    
    echo "Started $experiment_name with PID $!"
    echo "---"
}

# Main execution
if [ "$1" = "run" ]; then
    if [ -z "$2" ] || [ -z "$3" ]; then
        echo "Usage: $0 run <cuda_id> <run_id>"
        echo "Example: $0 run 0 1"
        exit 1
    fi
    
    cuda_id=$2
    run_id=$3
    
    echo "Running all inference ablation experiments on cuda:$cuda_id"
    echo "Base checkpoint: $BASE_CHECKPOINT"
    echo "Log directory: $LOG_DIR"
    echo "=================================="
    
    # Run all experiments
    run_experiment $cuda_id "baseline" "none" "none"
    run_experiment $cuda_id "drop_first_author" "drop_first" "none"
    run_experiment $cuda_id "drop_last_author" "drop_last" "none"
    run_experiment $cuda_id "keep_first_author" "keep_first" "none"
    run_experiment $cuda_id "keep_last_author" "keep_last" "none"
    run_experiment $cuda_id "drop_first_last_author" "drop_first_and_last" "none"
    run_experiment $cuda_id "drop_all_authors" "no author" "none"
    run_experiment $cuda_id "drop_topic" "none" "drop topic"
    run_experiment $cuda_id "drop_first_author_and_topic" "drop_first" "drop topic"
    run_experiment $cuda_id "drop_last_author_and_topic" "drop_last" "drop topic"
    run_experiment $cuda_id "keep_first_author_and_topic" "keep_first" "drop topic"
    run_experiment $cuda_id "keep_last_author_and_topic" "keep_last" "drop topic"
    
    echo "All experiments started. Check logs in: $LOG_DIR"
    echo "To monitor progress: tail -f $LOG_DIR/*.log"
    
elif [ "$1" = "single" ]; then
    if [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ] || [ -z "$6" ]; then
        echo "Usage: $0 single <cuda_id> <experiment_name> <author_dropping> <topic_dropping> <run_id>"
        echo "Example: $0 single 0 drop_first_author drop_first none 1"
        exit 1
    fi
    run_experiment $2 $3 $4 $5
    
else
    echo "Usage: $0 {run|single}"
    echo ""
    echo "Commands:"
    echo "  run <cuda_id> <run_id>     - Run all ablation experiments"
    echo "  single <cuda_id> <exp_name> <author_drop> <topic_drop> <run_id> - Run single experiment"
    echo ""
    echo "Examples:"
    echo "  $0 run 0 1                 # Run all experiments on cuda:0"
    echo "  $0 single 0 drop_first drop_first none 1  # Run single experiment"
    exit 1
fi