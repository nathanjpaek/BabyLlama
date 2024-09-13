import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    GPT2TokenizerFast,
    LlamaForCausalLM,
    LlamaConfig,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import Subset, ConcatDataset
from random import sample
from pathlib import Path
from babylm_dataset import BabylmDataset
import wandb

# Constants
LR = 2.5e-4
BATCH_SIZE = 32
SEQ_LENGTH = 128
TEMPERATURE = 2.0
ALPHA = 0.5
EVAL_SAMPLES = 8192

# Paths and model names
PATH = Path("./")
MODEL_NAME = 'Baby-Llama-58M'
MODEL_OUTPUT = PATH / 'models' / MODEL_NAME

tokenizer_path = PATH / "models/gpt-clean-16000.json"
tokenizer = GPT2TokenizerFast(tokenizer_file=str(tokenizer_path))
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"  # Set pad token

# Ensure the pad_token_id is properly set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Define paths for datasets
task_dataset_paths = {
    "childes": PATH / "data/babylm_10M_clean_2/childes.train",
    "bnc_spoken": PATH / "data/babylm_10M_clean_2/bnc_spoken.train",
    "gutenberg": PATH / "data/babylm_10M_clean_2/gutenberg.train",
    "open_subtitles": PATH / "data/babylm_10M_clean_2/open_subtitles.train",
    "simple_wiki": PATH / "data/babylm_10M_clean_2/simple_wiki.train",
    "switchboard": PATH / "data/babylm_10M_clean_2/switchboard.train",
}

# Load datasets
task_datasets = {
    task_name: BabylmDataset(str(dataset_path), SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True, single_file=True)
    for task_name, dataset_path in task_dataset_paths.items()
}

# Create evaluation dataset
full_eval_dataset = BabylmDataset(PATH / "data/babylm_dev_clean", SEQ_LENGTH, tokenizer=tokenizer, offset=0)
eval_indices = sample(range(len(full_eval_dataset)), EVAL_SAMPLES)
eval_dataset = Subset(full_eval_dataset, eval_indices)

# Model configuration
config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,  # Ensure vocab_size matches tokenizer's vocab size
    hidden_size=512,
    num_hidden_layers=16,
    intermediate_size=1024,
    num_attention_heads=8,
    bos_token_id=tokenizer.convert_tokens_to_ids("<s>"),
    eos_token_id=tokenizer.convert_tokens_to_ids("</s>"),
    pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),  # Set pad_token_id correctly
    max_position_embeddings=2 * SEQ_LENGTH,
)

# Initialize models
student = LlamaForCausalLM(config)
teacher1 = LlamaForCausalLM.from_pretrained(PATH / './models/Llama-360M')
teacher2 = GPT2LMHeadModel.from_pretrained(PATH / './models/GPT2-705M')
teachers = [teacher1, teacher2]

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Custom trainer with meta-learning and mixed precision
class MAMLTrainer(Trainer):
    def __init__(self, *args, teacher_models=None, task_datasets=None, maml_inner_lr=1e-3, maml_inner_steps=1, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teachers = teacher_models
        self.task_datasets = task_datasets
        self.maml_inner_lr = maml_inner_lr
        self.maml_inner_steps = maml_inner_steps
        self.alpha = alpha
        self.temperature = temperature
        self.scaler = GradScaler()  # Initialize GradScaler
        
        for teacher in self.teachers:
            teacher.to(self.model.device)
            teacher.eval()

    def inner_loop(self, model, task_dataset):
        adapted_model = type(model)(model.config).to(self.args.device)
        adapted_model.load_state_dict(model.state_dict())  # Copy weights from the original model
        inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.maml_inner_lr)
        task_dataloader = torch.utils.data.DataLoader(task_dataset, batch_size=self.args.per_device_train_batch_size)

        for step, batch in enumerate(task_dataloader):
            if step >= self.maml_inner_steps:
                break

            # Check if batch is a dictionary, otherwise handle it as a tensor
            if isinstance(batch, dict):
                inputs = {key: value.to(self.args.device) for key, value in batch.items()}
            else:
                # If it's a tensor, assume it's input_ids and prepare inputs accordingly
                inputs = {"input_ids": batch.to(self.args.device)}

            # Ensure "labels" are present for computing the loss
            if "labels" not in inputs:
                inputs["labels"] = inputs["input_ids"].clone()

            # Forward pass through the model with autocast for mixed precision
            with autocast():
                outputs = adapted_model(**inputs)
                loss = outputs.loss

            # Scale the loss for mixed precision
            self.scaler.scale(loss).backward()
            inner_optimizer.step()
            inner_optimizer.zero_grad()

        return adapted_model

    def compute_loss(self, model, inputs, return_outputs=False):
        task_name = sample(list(self.task_datasets.keys()), 1)[0]
        task_dataset = self.task_datasets[task_name]
        adapted_student = self.inner_loop(model, task_dataset)
        
        # Forward pass for student model
        with autocast():
            outputs_student = adapted_student(**inputs)
            student_loss = outputs_student.loss
            
            # Forward pass for teacher models and distillation loss
            with torch.no_grad():
                avg_teacher_logits = torch.stack([teacher(**inputs).logits for teacher in self.teachers]).mean(dim=0)
            
            distill_loss = nn.KLDivLoss(reduction="batchmean")(
                nn.functional.log_softmax(outputs_student.logits / self.temperature, dim=-1),
                nn.functional.softmax(avg_teacher_logits / self.temperature, dim=-1)
            ) * (self.temperature ** 2)
        
        # Combine student loss and distillation loss
        total_loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss
        return (total_loss, outputs_student) if return_outputs else total_loss

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Compute the loss with autocast for mixed precision
        with autocast():
            loss = self.compute_loss(model, inputs)
        
        # Scale the loss and backpropagate
        self.scaler.scale(loss).backward()

        # Return the scaled loss
        return loss.detach()

    def optimizer_step(self, optimizer, **kwargs):
        # Unscale gradients before clipping them
        self.scaler.unscale_(optimizer)

        # Optionally clip gradients
        if self.args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

        # Step the optimizer using the scaled gradients
        self.scaler.step(optimizer)
        self.scaler.update()


# Initialize wandb if needed
wandb_log = True
if wandb_log:
    wandb.login()
    wandb.init(project='babylm', name=MODEL_NAME)

# Custom MAML training arguments class
class MAMLTrainingArguments(TrainingArguments):
    def __init__(self, *args, maml_inner_lr=1e-3, maml_inner_steps=1, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.maml_inner_lr = maml_inner_lr
        self.maml_inner_steps = maml_inner_steps
        self.alpha = alpha
        self.temperature = temperature

# Define the training arguments for MAML
maml_training_args = MAMLTrainingArguments(
    output_dir=MODEL_OUTPUT,
    overwrite_output_dir=True,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    num_train_epochs=6,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=BATCH_SIZE,
    save_total_limit=1,
    report_to="wandb",
    warmup_steps=200,
    lr_scheduler_type="cosine",
    learning_rate=LR,
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    weight_decay=0.1,
    alpha=ALPHA,
    temperature=TEMPERATURE,
    maml_inner_lr=1e-3,
    maml_inner_steps=1,
    fp16=True,  # Enable mixed precision training
)

# Combine datasets and initialize trainer
train_dataset = ConcatDataset(list(task_datasets.values()))
maml_trainer = MAMLTrainer(
    model=student,
    args=maml_training_args,
    teacher_models=teachers,
    task_datasets=task_datasets,
    maml_inner_lr=maml_training_args.maml_inner_lr,
    maml_inner_steps=maml_training_args.maml_inner_steps,
    alpha=maml_training_args.alpha,
    temperature=maml_training_args.temperature,
    train_dataset=train_dataset,
    data_collator=data_collator,
    eval_dataset=eval_dataset,
)

# Train the model
maml_trainer.train()

# Save the model and tokenizer
maml_trainer.save_model(MODEL_OUTPUT)
tokenizer.save_pretrained(MODEL_OUTPUT)