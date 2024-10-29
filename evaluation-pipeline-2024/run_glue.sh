#!/bin/bash

# Usage: ./run_evaluate_glue_optimized.sh /path/to/model

MODELPATH=$1

# Check if model path is provided
if [ -z "$MODELPATH" ]; then
    echo "Usage: $0 /path/to/model"
    exit 1
fi

# List of GLUE tasks to evaluate. Modify the list as needed.
TASKS=("boolq" "cola" "mnli" "mrpc" "multirc" "qnli" "qqp" "rte" "sst2" "wsc")
# TASKS = ("cola")

# Directory to store all evaluation results
BASE_OUTPUT_DIR="results/eval"

# Create base output directory if it doesn't exist
mkdir -p "$BASE_OUTPUT_DIR"

for task in "${TASKS[@]}"; do
    echo "-------------------------------------------"
    echo "Evaluating task: $task"
    echo "-------------------------------------------"
    
    # Set output directory for the current task
    MODEL_NAME=$(basename "$MODELPATH")
    TASK_NAME=$(basename "$task")
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$MODEL_NAME/$task"
    mkdir -p "$OUTPUT_DIR"
    
    # Execute the evaluation script
    python evaluate_glue_optimized.py \
        "$MODELPATH" \
        "$task" \
        --batch_size 128 \
        --max_length 128 \
        --do_predict \
        --output_dir "$OUTPUT_DIR" \
        --num_proc 8 \
        --padding_side right \
        --device cuda
    
    echo "Evaluation for task '$task' completed. Results saved to '$OUTPUT_DIR'."
    echo ""
done

echo "All evaluations completed."
