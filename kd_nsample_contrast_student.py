import optuna
from transformers import (
    GPT2TokenizerFast,
    LlamaForCausalLM,
    LlamaConfig,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers import TrainerCallback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
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

teacher_dir1 = PATH / './models/Llama-360M'
teacher_dir2 = PATH / './models/GPT2-705M'

MODEL_NAME = f'Baby-Llama-58M-Contrastive_Student'
MODEL_OUTPUT = Path('./models') / MODEL_NAME
EVAL_SAMPLES = 8192

wandb_log = True

tokenizer_path = PATH / "models/gpt-clean-16000.json"
tokenizer = GPT2TokenizerFast(tokenizer_file=str(tokenizer_path))
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"

# Using a smaller dataset for faster hyperparameter tuning
train_dataset = BabylmDataset(PATH / "data/babylm_10M_clean", SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True)
# Use a smaller subset for hyperparameter tuning
eval_dataset = Subset(train_dataset, sample(range(len(train_dataset)), 1024))

tokenizer.model_max_length = SEQ_LENGTH

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
teacher1 = LlamaForCausalLM.from_pretrained(teacher_dir1)
teacher2 = GPT2LMHeadModel.from_pretrained(teacher_dir2)
teachers = [teacher1, teacher2]

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print(f'model num parameters: student = {student.num_parameters()}')
print(f'model num parameters: teacher1 = {teacher1.num_parameters()}')
print(f'model num parameters: teacher2 = {teacher2.num_parameters()}')

# N-sample contrastive loss function
def n_sample_contrastive_loss(student_logits, teacher_logits, temperature):
    # Normalize logits to unit norm
    student_normed = F.normalize(student_logits, dim=-1)
    teacher_normed = F.normalize(teacher_logits, dim=-1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(student_normed, teacher_normed.T)
    
    # Apply temperature
    similarity_matrix /= temperature
    
    # Calculate contrastive loss using cross-entropy
    batch_size = student_logits.size(0)
    labels = torch.arange(batch_size, device=student_logits.device)
    contrastive_loss = F.cross_entropy(similarity_matrix, labels)
    
    return contrastive_loss

# Custom TrainingArguments class to include contrastive weight
class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, contrastive_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight

# Custom Trainer to include distillation and n-sample contrastive loss
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_models=None, contrastive_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.teachers = teacher_models
        self.contrastive_weight = contrastive_weight
        for teacher in self.teachers:
            self._move_model_to_device(teacher, self.model.device)
            teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss

        with torch.no_grad():
            all_teacher_logits = []
            for teacher in self.teachers:
                outputs_teacher = teacher(**inputs)
                all_teacher_logits.append(outputs_teacher.logits)
            avg_teacher_logits = torch.stack(all_teacher_logits).mean(dim=0)

        assert outputs_student.logits.size() == avg_teacher_logits.size()

        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
            loss_function(
                F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
                F.softmax(avg_teacher_logits / self.args.temperature, dim=-1),
            )
            * (self.args.temperature ** 2)
        )

        # Compute n-sample contrastive loss
        contrastive_loss_value = n_sample_contrastive_loss(outputs_student.logits, avg_teacher_logits, self.args.temperature)

        # Combine student loss, distillation loss, and n-sample contrastive loss
        loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        total_loss = loss + self.contrastive_weight * contrastive_loss_value

        return (total_loss, outputs_student) if return_outputs else total_loss

# Custom pruning callback based on eval_loss
class HuggingFacePruningCallback(TrainerCallback):
    def __init__(self, trial, metric_name):
        super().__init__()
        self.trial = trial
        self.metric_name = metric_name

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        current_score = metrics.get(self.metric_name)
        if current_score is None:
            return
        # Report the metric score to Optuna
        self.trial.report(current_score, step=state.global_step)
        # Check if trial should be pruned
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()

# Optuna objective function with pruning and limited search space
def objective(trial):
    # Suggest contrastive weight value from a reduced range (0.1 to 0.5)
    contrastive_weight = trial.suggest_float('contrastive_weight', 0.1, 0.5)

    # Define the training arguments with fewer epochs for faster tuning
    training_args = DistillationTrainingArguments(
        output_dir=MODEL_OUTPUT,
        overwrite_output_dir=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        num_train_epochs=2,  # Fewer epochs for hyperparameter tuning
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
        contrastive_weight=contrastive_weight,  # Tune this using Optuna
    )

    # Initialize the trainer
    trainer = DistillationTrainer(
        student,
        training_args,
        teacher_models=teachers,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Enable custom pruning callback
    pruning_callback = HuggingFacePruningCallback(trial, "eval_loss")
    trainer.add_callback(pruning_callback)

    # Train the model
    trainer.train()

    # Evaluate the model and return eval_loss
    eval_metrics = trainer.evaluate()
    return eval_metrics["eval_loss"]

# Initialize Optuna and start tuning with early stopping and limited trials
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)  # Limit to 5 trials

# Get the best trial (best contrastive_weight)
best_trial = study.best_trial
print(f"Best trial: contrastive_weight={best_trial.params['contrastive_weight']}, eval_loss={best_trial.value}")

# Rerun the training for longer duration using full dataset with the best contrastive_weight
full_eval_dataset = BabylmDataset(PATH / "data/babylm_dev_clean", SEQ_LENGTH, tokenizer=tokenizer, offset=0)

training_args_full = DistillationTrainingArguments(
    output_dir=MODEL_OUTPUT,
    overwrite_output_dir=True,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    num_train_epochs=6,  # Longer training after finding the best hyperparameter
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
    contrastive_weight=best_trial.params['contrastive_weight'],  # Best contrastive weight
)

# Train again using the full dataset
trainer_full = DistillationTrainer(
    student,
    training_args_full,
    teacher_models=teachers,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=full_eval_dataset,
)

trainer_full.train()
