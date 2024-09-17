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
from torch.utils.data import Subset
from random import sample

from pathlib import Path
import wandb


#############
LR = 5e-5  # Reduced for fine-tuning
BATCH_SIZE = 16  # Reduced batch size for faster convergence
SEQ_LENGTH = 128

TEMPERATURE = 1.5  # Lower temperature for sharper logits during fine-tuning
ALPHA = 0.5  # Same alpha for balance between student and teacher loss
#############

PATH = Path("./")
teacher_student_dir = PATH / './models/TrainAll-2-Nsample-Contrastive_Student/checkpoint-24489'  # Same directory for both student and teacher

MODEL_NAME = f'Baby-Llama-58M-SelfDistill-Finetune'
MODEL_OUTPUT = Path('./models') /  MODEL_NAME
EVAL_SAMPLES = 1024  # Reduced sample size for faster evaluation

wandb_log = True

# Tokenizer setup
tokenizer_path = PATH / "models/gpt-clean-16000.json"
tokenizer = GPT2TokenizerFast(tokenizer_file=str(tokenizer_path))
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"

# Dataset setup
train_dataset = BabylmDataset(PATH / "data/babylm_10M_clean", SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True)
full_eval_dataset = BabylmDataset(PATH / "data/babylm_dev_clean", SEQ_LENGTH, tokenizer=tokenizer, offset=0)

eval_indices = sample(range(len(full_eval_dataset)), EVAL_SAMPLES)
eval_dataset = Subset(full_eval_dataset, eval_indices)

tokenizer.model_max_length = SEQ_LENGTH

# Load the model for both student and teacher
student = LlamaForCausalLM.from_pretrained(teacher_student_dir)
teacher = LlamaForCausalLM.from_pretrained(teacher_student_dir)
teachers = [teacher]

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

print(f'model num parameters: student = {student.num_parameters()}')
print(f'model num parameters: teacher = {teacher.num_parameters()}')


# Distillation Trainer setup (same as before but using self-distillation)
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
            self._move_model_to_device(teacher, self.model.device)
            teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        # Compute student output
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss

        # Compute teacher output
        with torch.no_grad():
            all_teacher_logits = []
            for teacher in self.teachers:
                outputs_teacher = teacher(**inputs)
                all_teacher_logits.append(outputs_teacher.logits)
            avg_teacher_logits = torch.stack(all_teacher_logits).mean(dim=0)

        # Soften probabilities and compute distillation loss
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


if wandb_log:
    wandb.login()
    wandb.init(project='babylm', name=MODEL_NAME)


training_args = DistillationTrainingArguments(
    output_dir=MODEL_OUTPUT,
    overwrite_output_dir=True,
    save_strategy="epoch",
    evaluation_strategy=None,
    num_train_epochs=3,  # Reduced for faster training
    gradient_accumulation_steps=2,  # Higher for larger effective batch size
    per_device_train_batch_size=BATCH_SIZE,
    save_total_limit=1,  # Set to zero to avoid saving too many checkpoints
    report_to="wandb",
    warmup_steps=100,  # Shortened warmup
    lr_scheduler_type="cosine",
    learning_rate=LR,
    logging_steps=20,
    fp16=True,  # Enable mixed precision for faster training
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    weight_decay=0.05,  # Adjusted for fine-tuning
    alpha=ALPHA,
    temperature=TEMPERATURE,
)

trainer = DistillationTrainer(
    student,
    training_args,
    teacher_models=teachers,  # Using the same model as teacher and student
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

trainer.save_model(MODEL_OUTPUT)
tokenizer.save_pretrained(MODEL_OUTPUT)
