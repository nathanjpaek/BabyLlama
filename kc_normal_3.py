from transformers import (
    GPT2TokenizerFast,
    LlamaForCausalLM,
    LlamaConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from random import sample
from pathlib import Path
import wandb

from babylm_dataset import BabylmDataset

#############
# Constants
#############
LR = 2.5e-4
BATCH_SIZE = 32
SEQ_LENGTH = 128

TEMPERATURE = 2.0
ALPHA = 0.5

#############
# Paths
#############
PATH = Path("./")
teacher_dir = PATH / './models/TrainAll-2-Nsample-Contrastive_Student/checkpoint-24489'
baby_llama_teacher_dir = PATH / './models/babyllama-58M-real'

MODEL_NAME = 'Baby-Llama-58M-NC-2'
MODEL_OUTPUT = Path('./models') / MODEL_NAME
EVAL_SAMPLES = 8192

wandb_log = True

# Load Tokenizer
tokenizer_path = PATH / "models/gpt-clean-16000.json"
tokenizer = GPT2TokenizerFast(tokenizer_file=str(tokenizer_path))
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"

# Load Datasets
train_dataset = BabylmDataset(PATH / "data/babylm_10M_clean", SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True)
full_eval_dataset = BabylmDataset(PATH / "data/babylm_dev_clean", SEQ_LENGTH, tokenizer=tokenizer, offset=0)

eval_indices = sample(range(len(full_eval_dataset)), EVAL_SAMPLES)
eval_dataset = Subset(full_eval_dataset, eval_indices)

tokenizer.model_max_length = SEQ_LENGTH

# Model Configuration
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

# Load the pretrained teacher model
teacher_student_model = LlamaForCausalLM.from_pretrained(teacher_dir)

# Set the student model to be the same as the teacher model for self-distillation
student = LlamaForCausalLM.from_pretrained(teacher_dir)  # The student is initialized with the pretrained teacher model

# Enable gradient checkpointing for memory optimization
student.gradient_checkpointing_enable()

# Freeze all layers except the final 2 transformer blocks for fine-tuning
for param in student.parameters():
    param.requires_grad = False  # Freeze all layers

# Unfreeze the last 2 transformer blocks
for layer in student.model.layers[-2:]:
    for param in layer.parameters():
        param.requires_grad = True  # Unfreeze the last 2 layers

# Additional guidance teacher model (Baby-Llama-58M)
baby_llama_teacher = LlamaForCausalLM.from_pretrained(baby_llama_teacher_dir)
teachers = [teacher_student_model, baby_llama_teacher]

# Enable gradient checkpointing for teacher models
for teacher in teachers:
    teacher.gradient_checkpointing_enable()

# Data Collator for Language Modeling with dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, pad_to_multiple_of=None  # Enable dynamic padding
)

print(f'Model num parameters: student = {student.num_parameters()}')
print(f'Model num parameters: teacher_student = {teacher_student_model.num_parameters()}')
print(f'Model num parameters: baby_llama_teacher = {baby_llama_teacher.num_parameters()}')

#############
# Custom Trainer for Distillation
#############
class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_models=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teachers = teacher_models
        for teacher in self.teachers:
            # Move each teacher to the same device as the student
            self._move_model_to_device(teacher, self.model.device)
            teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        # Compute student output
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss

        # Compute teacher output (with guidance from both teachers)
        with torch.no_grad():
            all_teacher_logits = []
            for teacher in self.teachers:
                outputs_teacher = teacher(**inputs)
                all_teacher_logits.append(outputs_teacher.logits)
            avg_teacher_logits = torch.stack(all_teacher_logits).mean(dim=0)

        # Ensure sizes match
        assert outputs_student.logits.size() == avg_teacher_logits.size()

        # Compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
            loss_function(
                F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
                F.softmax(avg_teacher_logits / self.args.temperature, dim=-1),
            )
            * (self.args.temperature ** 2)
        )
        # Return weighted student loss
        loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss

# Initialize Weights and Biases (optional)
if wandb_log:
    wandb.login()
    wandb.init(project='babylm', name=MODEL_NAME)

#############
# Training Arguments with Optimizations
#############
training_args = DistillationTrainingArguments(
    output_dir=MODEL_OUTPUT,
    overwrite_output_dir=True,
    save_strategy="epoch",  # Save checkpoints at the end of each epoch
    evaluation_strategy=None,  # Evaluate at the end of each epoch
    num_train_epochs=3,  # Reduced epochs for faster fine-tuning
    gradient_accumulation_steps=4,  # Increased for larger effective batch size
    per_device_train_batch_size=BATCH_SIZE,  # Keep batch size small per GPU
    save_total_limit=1,  # Keep only the most recent model checkpoint
    report_to="wandb",
    warmup_steps=100,  # Shorter warmup
    lr_scheduler_type="linear",  # Use linear learning rate scheduler
    learning_rate=LR,
    logging_steps=20,  # Log every 20 steps
    fp16=True,  # Enable mixed precision for faster training
    load_best_model_at_end=False,  # Load best model based on evaluation loss
    metric_for_best_model="eval_loss",  # Track evaluation loss for saving best model
    weight_decay=0.05,  # Lower weight decay for more gradual updates
    alpha=ALPHA,
    temperature=TEMPERATURE
)

# Distillation Trainer Setup
trainer = DistillationTrainer(
    student,
    training_args,
    teacher_models=teachers,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the Model
trainer.train()

# Save the trained student model and tokenizer
trainer.save_model(MODEL_OUTPUT)
tokenizer.save_pretrained(MODEL_OUTPUT)
