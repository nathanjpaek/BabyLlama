import os
import json
import pandas as pd

# Define the path to the results directory
base_path = "results/finetune"

# Initialize a dictionary to store results
results = {}

# Traverse each model in the results/lora directory
for model_folder in os.listdir(base_path):
    model_path = os.path.join(base_path, model_folder)
    
    # Check if it is a directory
    if os.path.isdir(model_path):
        model_results = []
        
        # Traverse each inner folder in the model folder
        for inner_folder in os.listdir(model_path):
            inner_path = os.path.join(model_path, inner_folder)
            
            # Look for eval_results.json inside each inner folder
            eval_file = os.path.join(inner_path, "eval_results.json")
            if os.path.isfile(eval_file):
                # Load JSON data
                with open(eval_file, "r") as f:
                    data = json.load(f)
                
                # Check for 'eval_accuracy' or 'eval_matthews_correlation'
                if 'eval_accuracy' in data:
                    result_value = data['eval_accuracy'] * 100
                elif 'eval_matthews_correlation' in data:
                    result_value = data['eval_matthews_correlation']
                else:
                    result_value = None  # If neither metric is found
                
                # Append the result with the inner folder name
                model_results.append((inner_folder, result_value))
        
        # Calculate the macro average for the model if results are available
        valid_results = [res[1] for res in model_results if res[1] is not None]
        macro_average = sum(valid_results) / len(valid_results) if valid_results else None
        
        # Store the results for the model
        results[model_folder] = {
            "results": model_results,
            "macro_average": macro_average
        }

# Convert results to a DataFrame for easy viewing and printing
final_results = []
for model, data in results.items():
    for inner_folder, result in data["results"]:
        final_results.append({
            "Model": model,
            "Inner Folder": inner_folder,
            "Result": result
        })
    final_results.append({
        "Model": model,
        "Inner Folder": "Macro Average",
        "Result": data["macro_average"]
    })

# Create a DataFrame and print it
df = pd.DataFrame(final_results)
print(df)
# Save the DataFrame to a CSV file
df.to_csv("model_results.csv", index=False)
