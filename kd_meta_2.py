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
from torch.utils.data import Subset, ConcatDataset
from random import sample
from pathlib import Path
from babylm_dataset import BabylmDataset
import wandb

#############
LR = 2.5e-4
BATCH_SIZE = 32
SEQ_LENGTH = 128
TEMPERATURE = 2.0
ALPHA = 0.5
EVAL_SAMPLES = 8192
#############

# Paths and model names
PATH = Path("./")
MODEL_NAME = 'Baby-Llama-58M'
MODEL_OUTPUT = PATH / 'models' / MODEL_NAME

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

# Load datasets
task_datasets = {}
for task_name, dataset_path in task_dataset_paths.items():
    task_datasets[task_name] = BabylmDataset(
        str(dataset_path), SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True, single_file=True
    )

# Create evaluation dataset
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

# Initialize models
student = LlamaForCausalLM(config)
teacher1 = LlamaForCausalLM.from_pretrained(PATH / './models/Llama-360M')
teacher2 = GPT2LMHeadModel.from_pretrained(PATH / './models/GPT2-705M')
teachers = [teacher1, teacher2]

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Custom trainer with meta-learning
class MAMLTrainer(Trainer):
    def __init__(self, *args, teacher_models=None, task_datasets=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teachers = teacher_models
        self.task_datasets = task_datasets
        for teacher in self.teachers:
            teacher.to(self.model.device)
            teacher.eval()

    def inner_loop(self, model, task_dataset):
        adapted_model = type(model)(model.config).to(self.args.device)
        adapted_model.load_state_dict(model.state_dict())  # Copy weights from the original model
        inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.args.maml_inner_lr)
        task_dataloader = torch.utils.data.DataLoader(task_dataset, batch_size=self.args.per_device_train_batch_size)

        for step, batch in enumerate(task_dataloader):
            if step >= self.args.maml_inner_steps:
                break
            batch = self._prepare_inputs(batch)
            outputs = adapted_model(**batch)
            loss = outputs.loss
            loss.backward()
            inner_optimizer.step()
            inner_optimizer.zero_grad()

        return adapted_model

    def compute_loss(self, model, inputs, return_outputs=False):
        task_name = sample(list(self.task_datasets.keys()), 1)[0]
        task_dataset = self.task_datasets[task_name]
        adapted_student = self.inner_loop(model, task_dataset)

        # Student outputs
        outputs_student = adapted_student(**inputs)
        student_loss = outputs_student.loss

        # Teacher outputs and distillation loss
        with torch.no_grad():
            avg_teacher_logits = torch.stack([teacher(**inputs).logits for teacher in self.teachers]).mean(dim=0)
        
        distill_loss = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
            F.softmax(avg_teacher_logits / self.args.temperature, dim=-1)
        ) * (self.args.temperature ** 2)

        total_loss = self.args.alpha * student_loss + (1 - self.args.alpha) * distill_loss

        return (total_loss, outputs_student) if return_outputs else total_loss

# Add this at the top of your script to ensure wandb_log is defined
wandb_log = True  # or False if you don't want to log to wandb

# Initialize wandb if needed
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
    maml_inner_lr=1e-3,  # This is the missing attribute
    maml_inner_steps=1,
)

# Combine datasets and initialize trainer
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

# Train the model
maml_trainer.train()

# Save the model and tokenizer
maml_trainer.save_model(MODEL_OUTPUT)
tokenizer.save_pretrained(MODEL_OUTPUT)
