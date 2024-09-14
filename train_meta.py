from transformers import (
    GPT2Config, GPT2LMHeadModel, 
    LlamaConfig, LlamaForCausalLM, 
    GPTJConfig, GPTJForCausalLM
)
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import GPT2TokenizerFast
from torch.utils.data import Subset, DataLoader, ConcatDataset

from random import sample, seed
import torch
from torch import nn
import torch.optim as optim
from pathlib import Path
import yaml
import argparse

from babylm_dataset import BabylmDataset


# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./config/llama-16M.yaml", help="Configuration file path")
parser.add_argument("--lr", type=float, default=None, help="Learning rate")
parser.add_argument("--model_name", type=str, default=None, help="Model name")
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Override config parameters if provided as command-line arguments
if args.lr:
    config['training']['lr'] = args.lr
if args.model_name:
    config['model']['name'] = args.model_name


# Tokenizer setup
SEQ_LENGTH = config['data']['seq_length']
tokenizer_path = config['data']['tokenizer_path']
tokenizer = GPT2TokenizerFast(tokenizer_file= str(tokenizer_path))
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"
tokenizer.model_max_length = SEQ_LENGTH

# Load datasets
task_datasets = {
    "childes": BabylmDataset(config['data']['train_path'], SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True),
    "bnc_spoken": BabylmDataset(config['data']['train_path'], SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True),
    "gutenberg": BabylmDataset(config['data']['train_path'], SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True),
    "open_subtitles": BabylmDataset(config['data']['train_path'], SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True),
    "simple_wiki": BabylmDataset(config['data']['train_path'], SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True),
    "switchboard": BabylmDataset(config['data']['train_path'], SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True),
}

# Create evaluation dataset
full_eval_dataset = BabylmDataset(config['data']['eval_path'], SEQ_LENGTH, tokenizer=tokenizer, offset=0)
seed(2023)  # Fix the same subset for all models
eval_indices = sample(range(len(full_eval_dataset)), config['data']['eval_samples'])
eval_dataset = Subset(full_eval_dataset, eval_indices)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Dynamic Model Configuration
if config['model']['type'] == "Llama":
    model_config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=2 * tokenizer.model_max_length,
        hidden_size=config['model']['hidden_size'],
        intermediate_size=config['model']['intermediate_size'],
        num_hidden_layers=config['model']['n_layer'],
        num_attention_heads=config['model']['n_head'],
        pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
    )
    model = LlamaForCausalLM(model_config)
elif config['model']['type'] == "GPT2":
    model_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=2 * tokenizer.model_max_length,
        n_embd=config['model']['hidden_size'],
        n_layer=config['model']['n_layer'],
        n_head=config['model']['n_head'],
        resid_pdrop=config['model']['resid_pdrop'],
        embd_pdrop=config['model']['embd_pdrop'],
        attn_pdrop=config['model']['attn_pdrop'],
        pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
    )
    model = GPT2LMHeadModel(model_config)
elif config['model']['type'] == "GPTJ":
    model_config = GPTJConfig(
        vocab_size=tokenizer.vocab_size,
        n_positions=2 * tokenizer.model_max_length,
        n_embd=config['model']['hidden_size'],
        n_layer=config['model']['n_layer'],
        n_head=config['model']['n_head'],
        resid_pdrop=config['model']['resid_pdrop'],
        embd_pdrop=config['model']['embd_pdrop'],
        attn_pdrop=config['model']['attn_pdrop'],
        pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
    )
    model = GPTJForCausalLM(model_config)

print(f'Model parameters = {model.num_parameters()}')


# Custom trainer with meta-learning
class MAMLTrainer(Trainer):
    def __init__(self, *args, task_datasets=None, maml_inner_lr=1e-3, maml_inner_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_datasets = task_datasets
        self.maml_inner_lr = maml_inner_lr  # Store these custom attributes
        self.maml_inner_steps = maml_inner_steps

    def inner_loop(self, model, task_dataset):
        adapted_model = type(model)(model.config).to(self.args.device)
        adapted_model.load_state_dict(model.state_dict())  # Copy weights from the original model
        inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.maml_inner_lr)  # Use self.maml_inner_lr here
        task_dataloader = DataLoader(task_dataset, batch_size=self.args.per_device_train_batch_size)

        for step, batch in enumerate(task_dataloader):
            if step >= self.maml_inner_steps:  # Use self.maml_inner_steps here
                break

            # Ensure batch is a dictionary and contains the necessary inputs
            if isinstance(batch, dict):
                inputs = {key: value.to(self.args.device) for key, value in batch.items()}
            elif isinstance(batch, (tuple, list)):
                inputs = {
                    "input_ids": batch[0].to(self.args.device),
                    "attention_mask": batch[1].to(self.args.device),
                    "labels": batch[2].to(self.args.device) if len(batch) > 2 else None
                }
            else:
                inputs = {"input_ids": batch.to(self.args.device)}

            # Ensure that "labels" are provided for computing the loss
            if "labels" not in inputs:
                inputs["labels"] = inputs["input_ids"].clone()

            # Forward pass
            with torch.no_grad():  # Disable gradient tracking to save memory
                outputs = adapted_model(**inputs)
            loss = outputs.loss

            loss.backward()
            inner_optimizer.step()
            inner_optimizer.zero_grad()

        return adapted_model

    def compute_loss(self, model, inputs, return_outputs=False):
        # Sample a task for meta-learning
        task_name = sample(list(self.task_datasets.keys()), 1)[0]
        task_dataset = self.task_datasets[task_name]
        adapted_model = self.inner_loop(model, task_dataset)

        # Forward pass with the adapted model
        outputs = adapted_model(**inputs)
        loss = outputs.loss

        del adapted_model  # Clear the adapted model
        torch.cuda.empty_cache()  # Free up any unused cached memory

        return (loss, outputs) if return_outputs else loss


# Custom MAML training arguments class
class MAMLTrainingArguments(TrainingArguments):
    def __init__(self, *args, maml_inner_lr=1e-3, maml_inner_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.maml_inner_lr = maml_inner_lr
        self.maml_inner_steps = maml_inner_steps


# Define the training arguments for MAML
training_args = MAMLTrainingArguments(
    fp16=True,
    output_dir=config['logging']['output_dir'],
    overwrite_output_dir=True,
    save_strategy="epoch",
    evaluation_strategy=None,
    num_train_epochs=config['training']['num_epochs'],
    gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
    per_device_train_batch_size=config['training']['batch_size'],
    learning_rate=config['training']['lr'],
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    warmup_steps=config['training']['warmup_steps'],
    lr_scheduler_type="cosine",
    maml_inner_lr=1e-3,  # Inner loop learning rate
    maml_inner_steps=1,   # Number of inner loop steps
)

# Initialize the MAML trainer
maml_trainer = MAMLTrainer(
    model=model,
    args=training_args,
    task_datasets=task_datasets,  # Pass the task datasets for meta-learning
    data_collator=data_collator,
    train_dataset=ConcatDataset(list(task_datasets.values())),
    eval_dataset=eval_dataset,
)

if __name__ == "__main__":
    if config['logging']['wandb']:
        import wandb
        wandb.login()
        wandb.init(project=config['logging']['project'], name=config['model']['name'], config=config)

    maml_trainer.train()
    maml_trainer.save_model(config['logging']['output_dir'])
    tokenizer.save_pretrained(config['logging']['output_dir'])
