import argparse
import os
import numpy as np
from copy import deepcopy

import torch
from torch.utils.data import DataLoader

import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from transformers.trainer_utils import set_seed
from tqdm import tqdm

def get_column_names(taskname):
    input_columns = []
    if taskname == "boolq":
        input_columns.extend(["question", "passage"])
    elif taskname in ("cola", "sst2"):
        input_columns.append("sentence")
    elif taskname in ("mnli", "mnli-mm"):
        input_columns.extend(["premise", "hypothesis"])
    elif taskname in ("mrpc", "rte"):
        input_columns.extend(["sentence1", "sentence2"])
    elif taskname == "multirc":
        input_columns.extend(["paragraph", "question_and_answer"])
    elif taskname == "qnli":
        input_columns.extend(["question", "sentence"])
    elif taskname == "qqp":
        input_columns.extend(["question1", "question2"])
    elif taskname == "wsc":
        input_columns.extend(["text", "span1_and_span2_text"])

    columns_to_remove = deepcopy(input_columns)
    columns_to_remove.append("idx")
    return (input_columns, columns_to_remove)


def load_tokenizer(tokenizer_path, padding_side):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, padding_side=padding_side, use_fast=True, trust_remote_code=True
    )
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def tokenize_fn(examples, tokenizer, input_columns=["sentence"], max_length=128):
    if len(input_columns) == 1:
        return tokenizer(examples[input_columns[0]], truncation=True, max_length=max_length)
    elif len(input_columns) == 2:
        return tokenizer(
            examples[input_columns[0]],
            examples[input_columns[1]],
            truncation=True,
            max_length=max_length
        )
    else:
        raise ValueError(f"Bad number of input_columns: {len(input_columns)}")


def collate_fn(examples, tokenizer):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


def main():
    parser = argparse.ArgumentParser(description="Optimized Evaluation of a pre-trained model on a GLUE task.")
    parser.add_argument("model_path", type=str, help="Path to the pre-trained model.")
    parser.add_argument("task", type=str, help="GLUE task name.", choices=[
        "boolq", "cola", "mnli", "mnli-mm", "mrpc", "multirc", "qnli", "qqp", "rte", "sst2", "wsc"
    ])
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation. Adjust based on your GPU memory.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run evaluation on.")
    parser.add_argument("--padding_side", type=str, default="right", choices=["left", "right"], help="Padding side for tokenizer.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the tokenizer. Defaults to `model_path`.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results. Defaults to `results/eval/model_name/task_name/`.")
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
    tokenizer_path = args.model_path if args.tokenizer_path is None else args.tokenizer_path
    tokenizer = load_tokenizer(tokenizer_path, args.padding_side)

    # Load dataset
    data_files = {
        "validation": f"evaluation_data/glue_filtered/{args.task}.valid.jsonl"
    }
    if args.task != "mnli-mm":
        data_files["train"] = f"evaluation_data/glue_filtered/{args.task}.train.jsonl"

    dataset = load_dataset("json", data_files=data_files, cache_dir="./cache", download_mode="reuse_dataset_if_exists")

    # Preprocess dataset based on task
    if args.task == "multirc":
        dataset = dataset.map(
            lambda example: {
                'question_and_answer': f"{example['question']} {example['answer']}"
            },
            remove_columns=['question', 'answer'],
            num_proc=args.num_proc
        )
    elif args.task == "wsc":
        dataset = dataset.map(
            lambda example: {
                'span1_and_span2_text': f"Does \"{example['span2_text']}\" refer to \"{example['span1_text']}\"?"
            },
            remove_columns=['span1_text', 'span2_text'],
            num_proc=args.num_proc
        )

    # Determine the metric set
    taskset = "super_glue" if args.task in ("boolq", "multirc", "wsc") else "glue"
    if args.task == "multirc":
        metric = evaluate.load(taskset, "wsc")  # Use accuracy for multirc similar to WSC
    else:
        metric = evaluate.load(taskset, args.task)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Set output directory
    if args.output_dir is None:
        task_basename = os.path.splitext(os.path.basename(args.task))[0]
        model_basename = os.path.basename(os.path.normpath(args.model_path))
        output_dir = f"results/eval/{model_basename}/{task_basename}/"
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    input_columns, columns_to_remove = get_column_names(args.task)

    # Tokenize dataset with parallel processing
    tokenized_dataset = dataset["validation"].map(
        tokenize_fn,
        batched=True,
        remove_columns=columns_to_remove,
        num_proc=args.num_proc,
        fn_kwargs={
            "tokenizer": tokenizer,
            "input_columns": input_columns,
            "max_length": args.max_length
        },
        desc="Tokenizing dataset"
    )
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    if args.task == "mnli-mm":
        num_labels = len(np.unique(tokenized_dataset["labels"]))
    else:
        num_labels = len(np.unique(dataset["validation"]["label"]))

    # Load model with mixed precision
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=num_labels,
        trust_remote_code=True,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    # Prepare Trainer with mixed precision and optimized settings
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="no",  # Disable automatic evaluation
        save_strategy="no",        # No checkpoints needed
        logging_strategy="epoch",
        load_best_model_at_end=False,
        dataloader_num_workers=args.num_proc,
        fp16=True if args.device == "cuda" else False,  # Enable mixed precision if using GPU
        disable_tqdm=False,  # Enable progress bars
        no_cuda=(args.device != "cuda"),
        dataloader_pin_memory=True,  # Pin memory for faster data transfer
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        eval_dataset=tokenized_dataset,
        data_collator=lambda x: collate_fn(x, tokenizer),
        args=training_args
    )

    # Move model to evaluation mode
    model.eval()

    # Run evaluation with mixed precision and no gradient computations
    print(f"Evaluating model on task: {args.task}")
    with torch.no_grad():
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
