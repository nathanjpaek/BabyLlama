import os
import json
import csv

# Directory to check
results_dir = "/evaluation-pipeline-2024/results/blimp"

# Initialize the list for storing results
table = [["folder_name", "blimp_filtered_acc", "blimp_supplement_acc"]]

# Walk through all folders in the directory
for folder in os.listdir(results_dir):
    folder_path = os.path.join(results_dir, folder)
    
    if os.path.isdir(folder_path):
        json_file = os.path.join(folder_path, "blimp_results.json")
        
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Initialize variables for storing the acc values
                blimp_filtered_acc = None
                blimp_supplement_acc = None
                
                if "blimp_filtered" in data["results"]:
                    blimp_filtered_acc = data["results"]["blimp_filtered"]["acc,none"]
                
                if "blimp_supplement" in data["results"]:
                    blimp_supplement_acc = data["results"]["blimp_supplement"]["acc,none"]
                
                # Append the results to the table if both values exist
                if blimp_filtered_acc is not None and blimp_supplement_acc is not None:
                    table.append([folder, blimp_filtered_acc, blimp_supplement_acc])

# Print out the table
for row in table:
    print(row)

# Save table as a CSV file
csv_file = "/evaluation-pipeline-2024/results/blimp_results_summary.csv"
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(table)

print(f"Results saved to {csv_file}")
