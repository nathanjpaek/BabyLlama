from transformers import (
    GPT2Config, GPT2LMHeadModel, 
    LlamaConfig, LlamaForCausalLM, 
    GPTJConfig, GPTJForCausalLM
)
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import GPT2TokenizerFast
from torch.utils.data import Subset
from random import sample, seed
from pathlib import Path
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from babylm_dataset import BabylmDataset


# Contrastive Loss Function
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)
        
        # Create similarity matrix
        similarity_matrix = torch.matmul(z_i, z_j.T) / self.temperature
        
        # Create positive and negative labels
        positive_labels = torch.arange(batch_size).cuda()
        
        # Compute loss (contrastive loss for positive and negative pairs)
        loss = F.cross_entropy(similarity_matrix, positive_labels)
        return loss


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./config/llama-16M.yaml", help="Configuration file path")
parser.add_argument("--lr", type=float, default=None, help="Learning rate")
parser.add_argument("--model_name", type=str, default=None, help="Model name")
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)


# Override config parameters if provided as command-line arguments
if args.lr:
    config['training']['lr'] = args.lr
if args.model_name:
    config['model']['name'] = args.model_name


SEQ_LENGTH = config['data']['seq_length']

tokenizer_path = config['data']['tokenizer_path']
tokenizer = GPT2TokenizerFast(tokenizer_file= str(tokenizer_path))
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"

# in the original code I had random_chunk = False
# random_chunk=True is expected to improve the model performance a bit
train_dataset = BabylmDataset(config['data']['train_path'], SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True)
full_eval_dataset = BabylmDataset(config['data']['eval_path'], SEQ_LENGTH, tokenizer=tokenizer, offset=0)

seed(2023) # we fix the same subset for all models
eval_indices = sample(range(len(full_eval_dataset)), config['data']['eval_samples'])
eval_dataset = Subset(full_eval_dataset, eval_indices)

# We tokenize the whole dataset and then set the max length
tokenizer.model_max_length = SEQ_LENGTH

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Dynamic Model Configuration
if config['model']['type'] == "Llama":
    model_config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=2*tokenizer.model_max_length,
        hidden_size=config['model']['hidden_size'],
        intermediate_size=config['model']['intermediate_size'],
        num_hidden_layers=config['model']['n_layer'],
        num_attention_heads=config['model']['n_head'],
        tie_word_embeddings=config['model'].get('tie_word_embeddings', False),
        pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
    )
    model = LlamaForCausalLM(model_config)
elif config['model']['type'] == "GPT2":
    model_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=2*tokenizer.model_max_length,
        n_embd=config['model']['hidden_size'],
        n_layer=config['model']['n_layer'],
        n_head=config['model']['n_head'],
        resid_pdrop = config['model']['resid_pdrop'],
        embd_pdrop = config['model']['embd_pdrop'],
        attn_pdrop = config['model']['attn_pdrop'],
        pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
    )
    model = GPT2LMHeadModel(model_config)
elif config['model']['type'] == "GPTJ":
    model_config = GPTJConfig(
        vocab_size=tokenizer.vocab_size,
        n_positions=2*tokenizer.model_max_length,
        n_embd=config['model']['hidden_size'],
        n_layer=config['model']['n_layer'],
        n_head=config['model']['n_head'],
        resid_pdrop = config['model']['resid_pdrop'],
        embd_pdrop = config['model']['embd_pdrop'],
        attn_pdrop = config['model']['attn_pdrop'],
        tie_word_embeddings=config['model']['tie_word_embeddings'],
        pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
    )
    model = GPTJForCausalLM(model_config)

print(f'model parameters = {model.num_parameters()}')


output_dir = Path(config['logging']['output_dir']) / (config['model']['name'] + "_contrastive")
accumulation_steps = config['training']['gradient_accumulation_steps']
per_device_bsz = config['training']['batch_size'] // accumulation_steps

# Custom Trainer for Contrastive Learning
class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Split the logits along the sequence dimension
        half_seq_length = SEQ_LENGTH // 2
        z_i = logits[:, :half_seq_length, :]  # First half of the sequence
        z_j = logits[:, half_seq_length:, :]  # Second half of the sequence

        # Average the embeddings over the sequence dimension (to get (batch_size, hidden_dim))
        z_i = z_i.mean(dim=1)
        z_j = z_j.mean(dim=1)

        # Set temperature for contrastive loss
        contrastive_temperature = config['training'].get('contrastive_temperature', 0.07)  # Default temperature is 0.07
        
        # Contrastive loss computation
        contrastive_loss_fn = ContrastiveLoss(temperature=contrastive_temperature)
        loss = contrastive_loss_fn(z_i, z_j)

        return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    save_strategy = "epoch",
    evaluation_strategy = "epoch",
    num_train_epochs=config['training']['num_epochs'],
    gradient_accumulation_steps=accumulation_steps,
    per_device_train_batch_size=per_device_bsz,
    save_total_limit=1,  # Set to zero to avoid saving
    warmup_steps=config['training']['warmup_steps'], 
    lr_scheduler_type="cosine",
    learning_rate=float(config['training']['lr']),
    logging_steps=20,
    fp16=config['training']['fp16'],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    torch_compile = config['training'].get('torch_compile', False),
)

trainer = ContrastiveTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)


if __name__ == "__main__":

    if config['logging']['wandb']:
        import wandb
        wandb.login()
        wandb.init(project= config['logging']['project'], name=config['model']['name']+"_contrastive", config=config)

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
