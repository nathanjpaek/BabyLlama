from transformers import (
    GPT2Config, GPT2LMHeadModel, 
    LlamaConfig, LlamaForCausalLM, 
    GPTJConfig, GPTJForCausalLM
)
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import GPT2TokenizerFast
from torch.utils.data import Subset
from random import seed
from pathlib import Path
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from babylm_dataset import BabylmDataset

# N-sample Contrastive Loss Function
class NSampleContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, num_samples=5):
        super(NSampleContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.num_samples = num_samples  # number of positive augmentations for each sample

    def forward(self, z_i, z_j_samples):
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=-1)
        z_j_samples = [F.normalize(z_j, dim=-1) for z_j in z_j_samples]
        
        batch_size = z_i.size(0)

        # Compute similarities for positive pairs
        pos_similarities = [torch.matmul(z_i, z_j.T) / self.temperature for z_j in z_j_samples]
        
        # Average positive similarities across multiple augmented views
        pos_similarity_matrix = torch.stack(pos_similarities).mean(dim=0)

        # Generate labels (positive pairs are along the diagonal)
        positive_labels = torch.arange(batch_size).cuda()

        # Contrastive loss (using cross entropy)
        loss = F.cross_entropy(pos_similarity_matrix, positive_labels)
        return loss

# Custom Trainer for N-sample Contrastive Learning
class NSampleContrastiveTrainer(Trainer):
    def __init__(self, contrastive_temperature, n_samples, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_temperature = contrastive_temperature
        self.n_samples = n_samples  # number of positive augmentations

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get augmented views for contrastive learning (multiple augmentations for each sample)
        z_i = logits  # Original token embeddings
        z_j_samples = [self.get_augmented_views(logits) for _ in range(self.n_samples)]  # Augmented token embeddings
        
        # Average the embeddings over the sequence dimension (to get (batch_size, hidden_dim))
        z_i = z_i.mean(dim=1)
        z_j_samples = [z_j.mean(dim=1) for z_j in z_j_samples]

        # Compute contrastive loss for n-sample contrastive learning
        contrastive_loss_fn = NSampleContrastiveLoss(temperature=self.contrastive_temperature, num_samples=self.n_samples)
        contrastive_loss = contrastive_loss_fn(z_i, z_j_samples)

        # Compute the language modeling loss
        lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Prioritize language modeling loss over contrastive loss (alpha for weighting)
        alpha = 0.9  # 90% focus on language modeling
        total_loss = alpha * lm_loss + (1 - alpha) * contrastive_loss

        return (total_loss, outputs) if return_outputs else total_loss

    def get_augmented_views(self, logits):
        # Implement your augmentation here (e.g., dropout, masking, etc.)
        # This is a simple example applying random dropout
        return F.dropout(logits, p=0.1, training=True)  # Augment logits with dropout


# Grid Search Function
def grid_search_temperature(train_dataset, tokenizer, model, config, temperature_range):
    best_temp = None
    best_loss = float("inf")

    for temp in temperature_range:
        print(f"Testing with contrastive temperature: {temp}")

        # Setup trainer for this temperature
        training_args = TrainingArguments(
            output_dir="./temp_output",
            num_train_epochs=1,  # Only train for 1 epoch during grid search
            per_device_train_batch_size=4,  # Use smaller batch size during grid search
            gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
            save_strategy="no",  # No saving during grid search
            logging_steps=10,
            fp16=config['training']['fp16'],
        )

        trainer = NSampleContrastiveTrainer(
            contrastive_temperature=temp,
            n_samples=5,  # Set n-sample augmentations during grid search
            model=model,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            train_dataset=train_dataset,
        )

        # Train and compute final loss
        result = trainer.train()
        final_loss = np.mean(result.metrics['train_loss'])

        print(f"Temperature: {temp}, Loss: {final_loss}")
        if final_loss < best_loss:
            best_loss = final_loss
            best_temp = temp

    return best_temp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/llama-16M.yaml", help="Configuration file path")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--model_name", type=str, default=None, help="Model name")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume training")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

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

    train_dataset = BabylmDataset(config['data']['train_path'], SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True)

    tokenizer.model_max_length = SEQ_LENGTH

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

    print(f'Model parameters = {model.num_parameters()}')

    # Perform grid search on contrastive loss temperature
    temperature_range = np.arange(0.05, 0.10, 0.01)
    best_temperature = grid_search_temperature(train_dataset, tokenizer, model, config, temperature_range)

    print(f"Best contrastive temperature: {best_temperature}")

    # Full training with the best temperature
    output_dir = Path(config['logging']['output_dir']) / (config['model']['name'] + "_contrastive_nsample")
    accumulation_steps = config['training']['gradient_accumulation_steps']
    per_device_bsz = (config['training']['batch_size'] * 2) // accumulation_steps  # Increased batch size

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        save_strategy="epoch",  # Save checkpoints every epoch
        evaluation_strategy="no",  # No evaluation
        num_train_epochs=config['training']['num_epochs'],
        gradient_accumulation_steps=accumulation_steps,
        per_device_train_batch_size=per_device_bsz,
        save_total_limit=1,  # Keep only the latest checkpoint
        warmup_steps=config['training']['warmup_steps'], 
        lr_scheduler_type="cosine",
        learning_rate=float(config['training']['lr']),
        logging_steps=20,
        fp16=config['training']['fp16'],
        load_best_model_at_end=False,
        torch_compile=config['training'].get('torch_compile', False),
        resume_from_checkpoint=args.resume_from_checkpoint  # Resume from checkpoint if provided
    )

    trainer = NSampleContrastiveTrainer(
        contrastive_temperature=best_temperature,
        n_samples=5,  # Number of positive augmentations in full training
        model=model,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        train_dataset=train_dataset,
    )

    # Resume training if checkpoint exists
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
