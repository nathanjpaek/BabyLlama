#!/bin/bash

# run_evaluate_piqa_optimized.sh

# Usage: ./run_evaluate_piqa_optimized.sh /path/to/model

# Check if model path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/model"
    exit 1
fi

MODELPATH=$1

# Verify that Python script exists
PYTHON_SCRIPT="evaluate_piqa_optimized.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Python script '$PYTHON_SCRIPT' not found in the current directory."
    exit 1
fi

# Create base output directory
BASE_OUTPUT_DIR="results/eval_piqa"
mkdir -p "$BASE_OUTPUT_DIR"

# Set default parameters
BATCH_SIZE=128
MAX_LENGTH=128
NUM_PROC=8
DEVICE="cuda"  # Change to "cpu" if GPU is not available

# Define output directory based on model name
MODEL_NAME=$(basename "$MODELPATH")
OUTPUT_DIR="$BASE_OUTPUT_DIR/$MODEL_NAME"
mkdir -p "$OUTPUT_DIR"

echo "-------------------------------------------"
echo "Evaluating PIQA with model: $MODEL_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Max sequence length: $MAX_LENGTH"
echo "Number of processes: $NUM_PROC"
echo "Device: $DEVICE"
echo "-------------------------------------------"

# Execute the evaluation script
python "$PYTHON_SCRIPT" \
    "$MODELPATH" \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --do_predict \
    --output_dir "$OUTPUT_DIR" \
    --num_proc "$NUM_PROC" \
    --device "$DEVICE"

echo "-------------------------------------------"
echo "Evaluation for PIQA completed."
echo "Results saved to '$OUTPUT_DIR'."
echo "-------------------------------------------"
