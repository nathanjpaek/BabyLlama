#!/bin/bash

# List of models to process
models=(
    GPT2-18M-childes
    GPT2-18M-stories
    GPT2-44M-childes
    GPT2-44M-stories
    GPT2-97M-all
    GPT2-97M-g10
    GPT2-18M-all
    GPT2-18M-gutenberg
    GPT2-44M-all
    GPT2-44M-gutenberg
    GPT2-705M-all
    GPT2-97M-childes
    GPT2-97M-stories
)

# Directory where collect_results.py is located
SCRIPT_DIR="collect_results.py"  # Replace with the actual path if needed

# Navigate to the script directory
cd "$SCRIPT_DIR" || { echo "Script directory not found!"; exit 1; }

# Loop through each model and process
for model in "${models[@]}"; do
    echo "Processing model: $model"
    
    # Run collect_results.py for text-only predictions
    python collect_results.py "$model" --include_vision_tasks
    
    # Define the expected output filename
    output_gz="${model}_textonly_predictions.json.gz"
    output_json="${model}_predictions.json"
    
    # Check if the gzipped file exists
    if [ -f "$output_gz" ]; then
        # Unzip and save as .json
        gunzip -c "$output_gz" > "$output_json"
        echo "Saved unzipped predictions to $output_json"
        
        # Optionally, remove the gzipped file to save space
        # rm "$output_gz"
    else
        echo "Error: Expected output file $output_gz not found for model $model."
    fi
    
    echo "-------------------------------------------"
done

echo "All models have been processed."
