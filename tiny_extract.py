# Open the local dataset file
file_path = "TinyStoriesV2-GPT4-train.txt"

# Read the file and tokenize words
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Function to tokenize and split the text into words
words = text.split()

# Extract the first 10 million words
word_limit = 10_000_000
extracted_words = words[:word_limit]

# Join the extracted words back into a text
extracted_text = ' '.join(extracted_words)

# Save the extracted text into a .train file
output_file = "TinyStoriesV2-GPT4-10M.train"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(extracted_text)

# Confirm that the file has been saved
print(f"Extracted text saved as {output_file}")
