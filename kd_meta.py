from transformers import (
    GPT2TokenizerFast,
    LlamaForCausalLM,
    LlamaConfig,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Union, Any

from torch.utils.data import Subset, ConcatDataset
from random import sample
from pathlib import Path
import wandb
from babylm_dataset import BabylmDataset

#############
LR = 2.5e-4
BATCH_SIZE = 32
SEQ_LENGTH = 128

TEMPERATURE = 2.0
ALPHA = 0.5
#############

PATH = Path("./")
MODEL_NAME = f'Baby-Llama-58M'
MODEL_OUTPUT = Path('./models') / MODEL_NAME
EVAL_SAMPLES = 8192
wandb_log = True

# Tokenizer setup
tokenizer_path = PATH / "models/gpt-clean-16000.json"
tokenizer = GPT2TokenizerFast(tokenizer_file=str(tokenizer_path))
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"
tokenizer.model_max_length = SEQ_LENGTH

# Define paths for datasets in babylm_10M_clean
task_dataset_paths = {
    "childes": PATH / "data/babylm_10M_clean_2/childes.train",
    "bnc_spoken": PATH / "data/babylm_10M_clean_2/bnc_spoken.train",
    "gutenberg": PATH / "data/babylm_10M_clean_2/gutenberg.train",
    "open_subtitles": PATH / "data/babylm_10M_clean_2/open_subtitles.train",
    "simple_wiki": PATH / "data/babylm_10M_clean_2/simple_wiki.train",
    "switchboard": PATH / "data/babylm_10M_clean_2/switchboard.train",
}

# Load each dataset into separate tasks, without saving tokenized data
# Load each dataset into separate tasks
task_datasets = {}
for task_name, dataset_path in task_dataset_paths.items():
    # Dynamically load raw .train data from each file (pass single_file=True)
    task_datasets[task_name] = BabylmDataset(
        str(dataset_path),  # Load raw .train data from here
        SEQ_LENGTH,
        tokenizer=tokenizer,
        random_chunk=True,
        single_file=True  # Specify that each dataset_path is a single file
    )


# Check each task dataset for its length
for task_name, dataset in task_datasets.items():
    print(f"Task: {task_name}")
    print(f"Length: {len(dataset)}")

# For evaluation, load the evaluation dataset as usual
full_eval_dataset = BabylmDataset(PATH / "data/babylm_dev_clean", SEQ_LENGTH, tokenizer=tokenizer, offset=0)
eval_indices = sample(range(len(full_eval_dataset)), EVAL_SAMPLES)
eval_dataset = Subset(full_eval_dataset, eval_indices)

# Model configuration
config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=512,
    num_hidden_layers=16,
    intermediate_size=1024,
    num_attention_heads=8,
    bos_token_id=tokenizer.convert_tokens_to_ids("<s>"),
    eos_token_id=tokenizer.convert_tokens_to_ids("</s>"),
    pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
    max_position_embeddings=2 * SEQ_LENGTH,
)

student = LlamaForCausalLM(config)

teacher1 = LlamaForCausalLM.from_pretrained(PATH / './models/Llama-360M')
teacher2 = GPT2LMHeadModel.from_pretrained(PATH / './models/GPT2-705M')
teachers = [teacher1, teacher2]

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# MAML Trainer implementation
class MAMLTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, maml_inner_lr=1e-3, maml_inner_steps=1, **kwargs):
        # Store these extra arguments separately
        self.alpha = alpha
        self.temperature = temperature
        self.maml_inner_lr = maml_inner_lr
        self.maml_inner_steps = maml_inner_steps
        
        # Pass other arguments to the parent class
        super().__init__(*args, **kwargs)

class MAMLTrainer(Trainer):
    def __init__(self, *args, teacher_models=None, task_datasets=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teachers = teacher_models
        self.task_datasets = task_datasets
        for teacher in self.teachers:
            self._move_model_to_device(teacher, self.model.device)
            teacher.eval()
        self.scaler = GradScaler()  # Initialize the scaler for mixed precision

    def inner_loop(self, model, task_dataset):
        """
        Inner loop for MAML: Adapt model parameters for a specific task.
        """
        adapted_model = LlamaForCausalLM(model.config).to(self.model.device)
        adapted_model.load_state_dict(model.state_dict())  # Copy weights from the original model

        inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.args.maml_inner_lr)
        task_dataloader = torch.utils.data.DataLoader(task_dataset, batch_size=self.args.per_device_train_batch_size)

        for step, batch in enumerate(task_dataloader):
            if step >= self.args.maml_inner_steps:
                break

            if isinstance(batch, dict):
                inputs = {key: value.to(self.model.device) for key, value in batch.items()}
            elif isinstance(batch, (tuple, list)):
                inputs = {
                    "input_ids": batch[0].to(self.model.device),
                    "attention_mask": batch[1].to(self.model.device),
                    "labels": batch[2].to(self.model.device) if len(batch) > 2 else None
                }
            else:
                inputs = {"input_ids": batch.to(self.model.device)}

            if "labels" not in inputs:
                inputs["labels"] = inputs["input_ids"].clone()

            with autocast():
                outputs_student = adapted_model(**inputs)
                student_loss = outputs_student.loss

            if student_loss is None:
                raise ValueError("Model did not return a loss. Ensure that 'labels' are provided in the inputs.")

            inner_optimizer.zero_grad()

            # Backpropagation with scaled gradients
            self.scaler.scale(student_loss).backward()

            # Step optimizer with scaled gradients
            self.scaler.step(inner_optimizer)
            self.scaler.update()

        return adapted_model

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Compute loss with autocast for mixed precision
        with autocast():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss = loss / self.args.gradient_accumulation_steps

        # Backpropagation with scaled gradients
        self.scaler.scale(loss).backward()

        return loss.detach()

    def train(self, resume_from_checkpoint=None, trial=None):
        # Ensure that scaler is initialized for every training session
        return super().train(resume_from_checkpoint=resume_from_checkpoint, trial=trial)

    def optimizer_step(self, model, optimizer, scheduler):
        # Unscale the gradients before calling `clip_grad_norm_`
        self.scaler.unscale_(optimizer)

        # Clip the gradients if necessary
        if self.args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

        # Perform optimizer step using the scaler
        self.scaler.step(optimizer)
        self.scaler.update()

        # Update the learning rate
        scheduler.step()

# ... [rest of the code remains unchanged] ...

# Initialize wandb if needed
if wandb_log:
    wandb.login()
    wandb.init(project='babylm', name=MODEL_NAME)

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
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    weight_decay=0.1,
    alpha=ALPHA,
    temperature=TEMPERATURE,
    maml_inner_lr=1e-3,
    maml_inner_steps=1,
)

# Combine all task datasets into one dataset for training
train_dataset = ConcatDataset(list(task_datasets.values()))

maml_trainer = MAMLTrainer(
    model=student,
    args=maml_training_args,
    teacher_models=teachers,
    task_datasets=task_datasets,
    train_dataset=train_dataset,
    data_collator=data_collator,
    eval_dataset=eval_dataset,
)

# Train the model using meta-learning
maml_trainer.train()

# Save the trained model and tokenizer
maml_trainer.save_model(MODEL_OUTPUT)
tokenizer.save_pretrained(MODEL_OUTPUT)