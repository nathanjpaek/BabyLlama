import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torch.nn.functional as F
from transformers import (
    GPT2TokenizerFast,
    LlamaForCausalLM,
    LlamaConfig,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from babylm_dataset import BabylmDataset  # Custom dataset class
from pathlib import Path
from random import sample
import wandb
import itertools
import subprocess
import os

# Define constants and hyperparameter ranges
##########
LR = 2.5e-4
BATCH_SIZE = 16  # Reduced batch size for faster training
SEQ_LENGTH = 128
TEMPERATURES = [1.0, 1.5, 2.0]
ALPHAS = [0.3, 0.5, 0.7]
INNER_LRS = [1e-3, 5e-4, 1e-4]
INNER_STEPS = [1, 2, 3]
EVAL_SAMPLES = 1024  # Reduced evaluation samples

# Additional hyperparams for tuning
WEIGHT_DECAYS = [0.01, 0.05, 0.1]
LR_SCHEDULERS = ["linear", "cosine"]
GRAD_ACCUMULATION_STEPS = [1, 2]
##########

# Paths and model names
PATH = Path("./")
MODEL_BASE_NAME = 'Meta-Student-2'
MODEL_OUTPUT_BASE = PATH / 'models' / MODEL_BASE_NAME

original_dir = os.getcwd()  # Keep as string
eval_dir = os.path.join(original_dir, 'evaluation-pipeline-2024')


# Tokenizer setup
tokenizer_path = PATH / "models/gpt-clean-16000.json"
tokenizer = GPT2TokenizerFast(tokenizer_file=str(tokenizer_path))
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"
tokenizer.model_max_length = SEQ_LENGTH

# Define paths for datasets
task_dataset_paths = {
    "childes": PATH / "data/babylm_10M_clean/childes.train",
    "bnc_spoken": PATH / "data/babylm_10M_clean/bnc_spoken.train",
    "gutenberg": PATH / "data/babylm_10M_clean/gutenberg.train",
    "open_subtitles": PATH / "data/babylm_10M_clean/open_subtitles.train",
    "simple_wiki": PATH / "data/babylm_10M_clean/simple_wiki.train",
    "switchboard": PATH / "data/babylm_10M_clean/switchboard.train",
}

# Load and reduce datasets (using 10% of each dataset)
task_datasets = {}
for task_name, dataset_path in task_dataset_paths.items():
    dataset = BabylmDataset(
        str(dataset_path), SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True, single_file=True
    )
    # Use 10% of the dataset
    subset_size = int(0.1 * len(dataset))
    subset_indices = sample(range(len(dataset)), subset_size)
    task_datasets[task_name] = Subset(dataset, subset_indices)

# Create evaluation dataset
full_eval_dataset = BabylmDataset(PATH / "data/babylm_dev_clean", SEQ_LENGTH, tokenizer=tokenizer, offset=0)
eval_indices = sample(range(len(full_eval_dataset)), EVAL_SAMPLES)
eval_dataset = Subset(full_eval_dataset, eval_indices)

# Model configuration (reduced size)
config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=256,  # Reduced model size
    num_hidden_layers=4,
    intermediate_size=1024,
    num_attention_heads=4,
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

# Define the Reptile algorithm trainer
class ReptileTrainer(Trainer):
    def __init__(self, *args, teacher_models=None, task_datasets=None, inner_lr=1e-3, inner_steps=1, alpha=0.5, temperature=2.0, data_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teachers = teacher_models
        self.task_datasets = task_datasets
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.alpha = alpha
        self.temperature = temperature
        self.data_collator = data_collator  # Store the data_collator
        for teacher in self.teachers:
            teacher.to(self.model.device)
            teacher.eval()
    
    def inner_loop(self, model, task_dataset):
        adapted_model = type(model)(model.config).to(self.args.device)
        adapted_model.load_state_dict(model.state_dict())  # Copy weights from the original model
        inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        # Use the data_collator in the DataLoader
        task_dataloader = DataLoader(
            task_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator  # Add collate_fn
        )
    
        for step, batch in enumerate(task_dataloader):
            if step >= self.inner_steps:
                break
            # Ensure batch is correctly formatted
            inputs = {key: value.to(self.args.device) for key, value in batch.items()}
            outputs = adapted_model(**inputs)
            loss = outputs.loss
            loss.backward()
            inner_optimizer.step()
            inner_optimizer.zero_grad()
        return adapted_model
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Sample a task
        task_name = sample(list(self.task_datasets.keys()), 1)[0]
        task_dataset = self.task_datasets[task_name]
        # Inner loop adaptation
        adapted_model = self.inner_loop(model, task_dataset)
        # Compute loss on batch
        outputs_student = adapted_model(**inputs)
        student_loss = outputs_student.loss
        with torch.no_grad():
            avg_teacher_logits = torch.stack([teacher(**inputs).logits for teacher in self.teachers]).mean(dim=0)
        distill_loss = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(outputs_student.logits / self.temperature, dim=-1),
            F.softmax(avg_teacher_logits / self.temperature, dim=-1)
        ) * (self.temperature ** 2)
        total_loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss
        # Reptile meta-update
        meta_gradient = [p_s - p for p_s, p in zip(adapted_model.parameters(), model.parameters())]
        for p, g in zip(model.parameters(), meta_gradient):
            if p.grad is not None:
                p.grad.zero_()
            p.grad = g
        return (total_loss, outputs_student) if return_outputs else total_loss

# Hyperparameter search
hyperparameter_space = list(itertools.product(TEMPERATURES, ALPHAS, INNER_LRS, INNER_STEPS, WEIGHT_DECAYS, LR_SCHEDULERS, GRAD_ACCUMULATION_STEPS))

# Initialize wandb if needed
wandb_log = False  # Set to True if you want to use wandb
if wandb_log:
    wandb.login()

# Loop over hyperparameter combinations
for temperature, alpha, inner_lr, inner_steps, weight_decay, lr_scheduler, grad_acc_steps in hyperparameter_space:
    print(f"Training with temperature={temperature}, alpha={alpha}, inner_lr={inner_lr}, inner_steps={inner_steps}, weight_decay={weight_decay}, lr_scheduler={lr_scheduler}, grad_acc_steps={grad_acc_steps}")
    # Update model name
    MODEL_NAME = f"{MODEL_BASE_NAME}_temp{temperature}_alpha{alpha}_ilr{inner_lr}_isteps{inner_steps}_wd{weight_decay}_lrsched{lr_scheduler}_gacc{grad_acc_steps}"
    MODEL_OUTPUT = MODEL_OUTPUT_BASE / MODEL_NAME
    # Reset student model
    student = LlamaForCausalLM(config)
    student.train()
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT,
        overwrite_output_dir=True,
        num_train_epochs=2,  # Train for 2 epochs now
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR,
        logging_steps=20,
        save_steps=1000,
        save_total_limit=1,
        report_to="wandb" if wandb_log else [],
        logging_dir='./logs',
        weight_decay=weight_decay,  # Added weight_decay
        lr_scheduler_type=lr_scheduler,  # Added learning rate scheduler
        gradient_accumulation_steps=grad_acc_steps,  # Added gradient accumulation steps
    )
    if wandb_log:
        wandb.init(project='babylm', name=MODEL_NAME)
    # Initialize trainer
    trainer = ReptileTrainer(
        model=student,
        args=training_args,
        teacher_models=teachers,
        task_datasets=task_datasets,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        alpha=alpha,
        temperature=temperature,
        train_dataset=ConcatDataset(list(task_datasets.values())),
        data_collator=data_collator,  # Pass the data_collator here
    )
    # Train model
    trainer.train()
    # Save model
    trainer.save_model(MODEL_OUTPUT)
    tokenizer.save_pretrained(MODEL_OUTPUT)
    if wandb_log:
        wandb.finish()
    # Run evaluation using lm_eval
    MODEL_PATH = "../" + str(MODEL_OUTPUT)
    MODEL_BASENAME = MODEL_OUTPUT.name
    try:
        # Change to the evaluation directory
        os.chdir(eval_dir)

        # Build command
        eval_command = [
            "python", "-m", "lm_eval", "--model", "hf",
            "--model_args", f'pretrained={MODEL_PATH},backend="causal"',
            "--tasks", "blimp_filtered,blimp_supplement",
            "--device", "cuda:0" if torch.cuda.is_available() else "cpu",
            "--batch_size", "1",
            "--log_samples",
            "--output_path", f"results/blimp/{MODEL_BASENAME}/blimp_results.json"
        ]

        # Run evaluation
        print(f"Evaluating model {MODEL_BASENAME}")
        subprocess.run(eval_command)
    
    finally:
        # Change back to the original directory
        os.chdir(original_dir)
