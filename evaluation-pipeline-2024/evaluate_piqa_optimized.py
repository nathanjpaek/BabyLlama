#!/usr/bin/env python3
# evaluate_piqa_optimized.py

import argparse
import os
import numpy as np
from copy import deepcopy

import torch
from torch.utils.data import DataLoader

import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from transformers.trainer_utils import set_seed
from tqdm import tqdm


def get_piqa_features(example, tokenizer, max_length=128):
    """
    Tokenizes PIQA examples for multiple-choice evaluation.
    """
    # Each example has 'goal', 'sol1', 'sol2'
    choices = [example['sol1'], example['sol2']]
    encoding = tokenizer(
        [example['goal']] * len(choices),
        choices,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return encoding


def preprocess_piqa(dataset, tokenizer, max_length=128, num_proc=8):
    """
    Preprocesses the PIQA dataset for multiple-choice evaluation.
    """
    def preprocess_function(examples):
        input_encodings = []
        labels = []
        for goal, sol1, sol2 in zip(examples['goal'], examples['sol1'], examples['sol2']):
            choices = [sol1, sol2]
            encoding = tokenizer(
                [goal] * len(choices),
                choices,
                padding='max_length',
                truncation=True,
                max_length=max_length
            )
            input_encodings.append(encoding)
        return {
            'input_ids': [enc['input_ids'] for enc in input_encodings],
            'attention_mask': [enc['attention_mask'] for enc in input_encodings],
            'labels': examples['label']
        }

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing PIQA dataset"
    )
    return tokenized_dataset


def main():
    parser = argparse.ArgumentParser(description="Optimized Evaluation of a pre-trained model on the PIQA benchmark.")
    parser.add_argument("model_path", type=str, help="Path or identifier of the pre-trained model.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation. Adjust based on your GPU memory.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run evaluation on.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results. Defaults to `results/eval_piqa/<model_name>/`.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length for tokenization.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to save predictions to a file.")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes for data loading and tokenization.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load PIQA dataset
    print("Loading PIQA dataset...")
    dataset = load_dataset("piqa")

    # Preprocess dataset
    print("Preprocessing dataset...")
    tokenized_dataset = preprocess_piqa(
        dataset['validation'],
        tokenizer,
        max_length=args.max_length,
        num_proc=args.num_proc
    )
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Load model
    print("Loading model...")
    model = AutoModelForMultipleChoice.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    # Define evaluation metric
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    # Set output directory
    if args.output_dir is None:
        model_basename = os.path.basename(os.path.normpath(args.model_path))
        output_dir = f"results/eval_piqa/{model_basename}/"
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Prepare Trainer arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="no",  # Disable automatic evaluation
        save_strategy="no",        # No checkpoints needed
        logging_strategy="epoch",
        load_best_model_at_end=False,
        dataloader_num_workers=args.num_proc,
        fp16=True if args.device == "cuda" else False,  # Enable mixed precision if using GPU
        no_cuda=(args.device != "cuda"),
        dataloader_pin_memory=True,  # Pin memory for faster data transfer
        disable_tqdm=False,  # Enable progress bars
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        eval_dataset=tokenized_dataset,
        args=training_args
    )

    # Run evaluation
    print("Running evaluation...")
    metrics = trainer.evaluate()
    trainer.save_metrics("eval", metrics)
    print(f"Evaluation metrics saved to {output_dir}")
    print(metrics)

    # If do_predict, save predictions
    if args.do_predict:
        print("Generating predictions...")
        predictions_output = trainer.predict(tokenized_dataset)
        predictions = np.argmax(predictions_output.predictions, axis=1)

        output_predict_file = os.path.join(output_dir, "predictions.txt")
        with open(output_predict_file, "w") as writer:
            writer.write("index\tprediction\n")
            for index, item in enumerate(predictions):
                writer.write(f"{index}\t{item}\n")
        print(f"Predictions saved to {output_predict_file}")


if __name__ == "__main__":
    main()
