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
from torch.utils.data import  Subset
from random import sample
from torch.utils.data import ConcatDataset

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
MODEL_OUTPUT = Path('./models') /  MODEL_NAME
EVAL_SAMPLES = 8192
wandb_log = True

# Tokenizer setup
tokenizer_path = PATH / "models/gpt-clean-16000.json"
tokenizer = GPT2TokenizerFast(tokenizer_file= str(tokenizer_path))
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"

# Set random_chunk=True for performance boost
tokenizer.model_max_length = SEQ_LENGTH

# Define paths for datasets as individual tasks
task_dataset_paths = {
    "childes": PATH / "data/babylm_10M_clean_2/childes.train",
    "bnc_spoken": PATH / "data/babylm_10M_clean_2/bnc_spoken.train",
    "gutenberg": PATH / "data/babylm_10M_clean_2/gutenberg.train",
    "open_subtitles": PATH / "data/babylm_10M_clean_2/open_subtitles.train",
    "simple_wiki": PATH / "data/babylm_10M_clean_2/simple_wiki.train",
    "switchboard": PATH / "data/babylm_10M_clean_2/switchboard.train",
}

# Define the tokenized directory
tokenized_dir = PATH / "data/babylm_10M_clean_2/tokenized"
tokenized_dir.mkdir(parents=True, exist_ok=True)  # Ensure it exists
# Load each dataset into separate tasks
task_datasets = {}
for task_name, dataset_path in task_dataset_paths.items():
    # Load each dataset with the correct dataset_path, not tokenized_dir
    task_datasets[task_name] = BabylmDataset(str(dataset_path), SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True)

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
        # Store alpha and temperature as class attributes
        self.alpha = alpha
        self.temperature = temperature
        self.maml_inner_lr = maml_inner_lr
        self.maml_inner_steps = maml_inner_steps

        # Pass remaining arguments to the base TrainingArguments class
        super().__init__(*args, **kwargs)

class MAMLTrainer(Trainer):
    def __init__(self, *args, teacher_models=None, task_datasets=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teachers = teacher_models
        self.task_datasets = task_datasets
        for teacher in self.teachers:
            self._move_model_to_device(teacher, self.model.device)
            teacher.eval()

    def inner_loop(self, model, task_dataset):
        """
        Inner loop for MAML: Adapt model parameters for a specific task.
        """
        adapted_model = LlamaForCausalLM.from_config(model.config).to(self.model.device)
        adapted_model.load_state_dict(model.state_dict())
        
        inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.args.maml_inner_lr)
        
        task_dataloader = torch.utils.data.DataLoader(task_dataset, batch_size=self.args.per_device_train_batch_size)
        
        for step, batch in enumerate(task_dataloader):
            if step >= self.args.maml_inner_steps:
                break
            outputs_student = adapted_model(**batch)
            student_loss = outputs_student.loss
            inner_optimizer.zero_grad()
            student_loss.backward()
            inner_optimizer.step()

        return adapted_model

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Outer loop: Compute meta-loss across tasks.
        """
        # Randomly select a task dataset for the inner loop
        task_name = sample(list(self.task_datasets.keys()), 1)[0]
        task_dataset = self.task_datasets[task_name]
        adapted_student = self.inner_loop(model, task_dataset)

        # Compute adapted student outputs
        outputs_student = adapted_student(**inputs)
        student_loss = outputs_student.loss

        # Compute teacher output for distillation
        with torch.no_grad():
            all_teacher_logits = []
            for teacher in self.teachers:
                outputs_teacher = teacher(**inputs)
                all_teacher_logits.append(outputs_teacher.logits)
            avg_teacher_logits = torch.stack(all_teacher_logits).mean(dim=0)

        # Distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
            loss_function(
                F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
                F.softmax(avg_teacher_logits / self.args.temperature, dim=-1),
            ) * (self.args.temperature ** 2)
        )

        # Combine student loss and distillation loss
        loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        return loss

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
    maml_inner_lr=1e-3,  # Inner loop learning rate
    maml_inner_steps=1,   # Inner loop step count
)

# Combine all task datasets into one dataset for training
train_dataset = ConcatDataset(list(task_datasets.values()))

maml_trainer = MAMLTrainer(
    model=student,
    args=maml_training_args,
    teacher_models=teachers,
    task_datasets=task_datasets,  # Pass all task-specific datasets for meta-learning
    train_dataset=train_dataset,  # Combine all tasks into a single training dataset
    data_collator=data_collator,
    eval_dataset=eval_dataset,
)

# Train the model using meta-learning
maml_trainer.train()

# Save the trained model and tokenizer
maml_trainer.save_model(MODEL_OUTPUT)
tokenizer.save_pretrained(MODEL_OUTPUT)
