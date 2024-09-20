from transformers import LlamaForCausalLM, GPT2TokenizerFast, GPT2LMHeadModel
import torch
from pathlib import Path

PATH = Path("./")
teacher_student_dir = PATH / './models/GPT2-705M'  # Same directory for both student and teacher

# Load the tokenizer and pre-trained model
tokenizer_path = PATH / "models/gpt-clean-16000.json"
tokenizer = GPT2TokenizerFast(tokenizer_file=str(tokenizer_path))
model = GPT2LMHeadModel.from_pretrained(teacher_student_dir)

# Enable gradient checkpointing to reduce memory usage
model.gradient_checkpointing_enable()

# Set tokenizer tokens (in case they aren't already)
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"

# Function to generate output and self-critique
# Function to generate output and self-critique
def generate_with_self_critique(prompt, critique_prompt_template):
    # Generate text from the model
    inputs = tokenizer(prompt, return_tensors="pt")
    output_ids = model.generate(inputs['input_ids'], max_new_tokens=100)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Generate critique of the output
    critique_prompt = critique_prompt_template.format(generated_text)
    critique_inputs = tokenizer(critique_prompt, return_tensors="pt")
    critique_output_ids = model.generate(critique_inputs['input_ids'], max_new_tokens=50)  # Use max_new_tokens
    critique = tokenizer.decode(critique_output_ids[0], skip_special_tokens=True)
    
    return generated_text, critique


# Function to filter output based on critique
def filter_based_on_critique(generated_text, critique, thresholds={"coherence": "good", "grammar": "correct"}):
    # You can adjust the filter logic based on critique content
    if "not coherent" in critique or "grammatical errors" in critique:
        print(f"Discarding low-quality output: {generated_text}")
        return False
    return True

# Save high-quality synthetic data to a .train file
def save_synthetic_data_to_file(synthetic_data, output_path):
    with open(output_path, 'w') as f:
        for entry in synthetic_data:
            f.write(entry + '\n')  # Save each entry in a new line, as in a .train file

# Generate high-quality synthetic data with critique and filtering
def generate_synthetic_data_with_self_critique(prompts, critique_prompt_template, output_path, num_samples=5):
    high_quality_outputs = []
    
    for prompt in prompts:
        for _ in range(num_samples):
            generated_text, critique = generate_with_self_critique(prompt, critique_prompt_template)
            print(f"Generated text: {generated_text}")
            print(f"Self-critique: {critique}")
            
            if filter_based_on_critique(generated_text, critique):
                high_quality_outputs.append(generated_text)
    
    # Save the high-quality outputs to a .train file
    save_synthetic_data_to_file(high_quality_outputs, output_path)
    print(f"High-quality synthetic data saved to {output_path}")

# Sample usage
prompts = [
    "Once upon a time, in a distant land, there was a great king.",
    "The scientific method is the cornerstone of modern research.",
]

# Critique template - the model will critique its own output
critique_prompt_template = "Critique this text for coherence and grammar: '{}'. Is it coherent and grammatically correct?"

# Define output file path in the "data" folder
output_path = PATH / './data/synthetic_data.train'  

# Generate high-quality synthetic data and save it to a .train file
generate_synthetic_data_with_self_critique(prompts, critique_prompt_template, output_path)
