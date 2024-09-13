import os
import torch
from torch.utils.data import Dataset
from random import randrange
from pathlib import Path

class BabylmDataset(Dataset):
    def __init__(self, data_dir: str, seq_length: int, tokenizer, offset: int = 0, random_chunk: bool = False, tokenized_dir_override: str = None):
        self.seq_length = seq_length
        self.offset = offset
        self.tokenizer = tokenizer
        self.random_chunk = random_chunk

        tokenizer_name = tokenizer.__class__.__name__

        # If tokenized_dir_override is provided, save the tokenized file there
        if tokenized_dir_override:
            tokenized_dir = Path(tokenized_dir_override)
            tokenized_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
            tokenized_file = tokenized_dir / f"tokenized_{tokenizer_name}_{tokenizer.vocab_size}.pt"
        else:
            tokenized_file = Path(os.path.join(data_dir, f"tokenized_{tokenizer_name}_{tokenizer.vocab_size}.pt"))

        # If tokenized data exists, load it; otherwise, process the raw data and tokenize it
        if tokenized_file.exists() and tokenized_file.stat().st_size > 0:
            print(f"Loading data from {tokenized_file}")
            self.data = torch.load(tokenized_file)
            print(f"Loaded data type: {type(self.data)}, shape: {self.data.shape if isinstance(self.data, torch.Tensor) else 'N/A'}")
        else:
            print(f"Tokenized data not found. Processing raw data from {data_dir}.")
            data = []
            src_files = [str(f) for f in Path(data_dir).glob("**/*")
                         if f.is_file() and not f.name.endswith(".DS_Store") and f.suffix in [".train", ".dev"]]

            for src_file in src_files:
                text = Path(src_file).read_text(encoding="utf-8")
                encoded = self.tokenizer.encode(text)
                print(f"🔥 {src_file}, len: {len(encoded)}")
                data.extend(encoded)

            self.data = torch.tensor(data)

            # Save tokenized data for future use
            print(f"Saving tokenized data to {tokenized_file}")
            torch.save(self.data, tokenized_file)

        # Final check to ensure data is loaded correctly
        if not isinstance(self.data, torch.Tensor) or len(self.data) == 0:
            raise ValueError(f"Failed to load valid data. self.data is of type {type(self.data)} and has length {len(self.data) if isinstance(self.data, torch.Tensor) else 'N/A'}.")

    def __len__(self):
        print(f"self.data has length: {len(self.data)}")
        if self.random_chunk:
            return len(self.data) // self.seq_length - 1
        else:
            return (len(self.data) - self.offset) // self.seq_length

    def __getitem__(self, i):
        if self.random_chunk:
            offset = randrange(self.seq_length)  # Sample random offset between 0 and seq_length-1
            return self.data[i * self.seq_length + offset:(i + 1) * self.seq_length + offset]
        else:
            return self.data[i * self.seq_length + self.offset:(i + 1) * self.seq_length + self.offset]
