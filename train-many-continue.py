from transformers import (
    GPT2Config, GPT2LMHeadModel, 
    LlamaConfig, LlamaForCausalLM, 
    GPTJConfig, GPTJForCausalLM,
    ElectraConfig, ElectraForCausalLM
)
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import GPT2TokenizerFast
from torch.utils.data import Subset
from random import sample, seed
from pathlib import Path
import yaml
import argparse

from babylm_dataset import BabylmDataset
from accelerate import Accelerator  # Import accelerate

# Initialize accelerator for CPU/GPU offloading
accelerator = Accelerator()

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
tokenizer = GPT2TokenizerFast(tokenizer_file=str(tokenizer_path))
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"

# In the original code, random_chunk = False
# random_chunk=True is expected to improve the model performance a bit
train_dataset = BabylmDataset(config['data']['train_path'], SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True)
full_eval_dataset = BabylmDataset(config['data']['eval_path'], SEQ_LENGTH, tokenizer=tokenizer, offset=0)

seed(2023)  # We fix the same subset for all models
eval_indices = sample(range(len(full_eval_dataset)), config['data']['eval_samples'])
eval_dataset = Subset(full_eval_dataset, eval_indices)

# We tokenize the whole dataset and then set the max length
tokenizer.model_max_length = SEQ_LENGTH

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

output_dir = Path(config['logging']['output_dir']) / config['model']['name']
accumulation_steps = config['training']['gradient_accumulation_steps']
per_device_bsz = config['training']['batch_size'] // accumulation_steps


# Dynamic Model Configuration
if config['model']['type'] == "Llama":
    model = LlamaForCausalLM.from_pretrained(output_dir)
elif config['model']['type'] == "GPT2":
    model = GPT2LMHeadModel.from_pretrained(output_dir)
elif config['model']['type'] == "GPTJ":
    model = GPTJForCausalLM.from_pretrained(output_dir)
elif config['model']['type'] == "Electra":
    model = ElectraForCausalLM.from_pretrained(output_dir)


# Offload to CPU if necessary
model = accelerator.prepare(model)

print(f'Model parameters = {model.num_parameters()}')
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=False,  # Do not overwrite the previous checkpoints
    save_strategy="epoch",
    evaluation_strategy="epoch",
    num_train_epochs=8,  # Continue from 4 epochs to 8 epochs total
    gradient_accumulation_steps=accumulation_steps,
    per_device_train_batch_size=per_device_bsz,
    save_total_limit=4,
    warmup_steps=config['training']['warmup_steps'], 
    lr_scheduler_type="cosine",
    learning_rate=float(config['training']['lr']),
    logging_steps=20,
    fp16=config['training']['fp16'],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    torch_compile=config['training'].get('torch_compile', False),
)


trainer = Trainer(
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
        wandb.init(project=config['logging']['project'], name=config['model']['name'], config=config)

    trainer.train(resume_from_checkpoint=True)  # Resume from last saved checkpoint
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

