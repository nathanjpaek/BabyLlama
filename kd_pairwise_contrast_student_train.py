import torch
from transformers import (
    GPT2TokenizerFast,
    LlamaForCausalLM,
    LlamaConfig,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from pathlib import Path
import wandb
import torch.nn as nn
import torch.nn.functional as F

from babylm_dataset import BabylmDataset

#############
LR = 2.5e-4
BATCH_SIZE = 32
SEQ_LENGTH = 128

TEMPERATURE = 2.0
ALPHA = 0.5
CONTRASTIVE_WEIGHT = 0.31395126385008854  # Best found contrastive weight
#############

PATH = Path("./")

teacher_dir1 = PATH / './models/Llama-360M'
teacher_dir2 = PATH / './models/GPT2-705M'

MODEL_NAME = f'TrainAll-Pairwise-Contrastive_Student'
MODEL_OUTPUT = Path('./models') / MODEL_NAME
EVAL_SAMPLES = 8192

wandb_log = True

tokenizer_path = PATH / "models/gpt-clean-16000.json"
tokenizer = GPT2TokenizerFast(tokenizer_file=str(tokenizer_path))
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"

# Using the full dataset for training
train_dataset = BabylmDataset(PATH / "data/babylm_10M_clean", SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True)
full_eval_dataset = BabylmDataset(PATH / "data/babylm_dev_clean", SEQ_LENGTH, tokenizer=tokenizer, offset=0)

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

# Contrastive loss function
def contrastive_loss(student_outputs, teacher_outputs, temperature):
    student_logits = F.normalize(student_outputs, dim=-1)
    teacher_logits = F.normalize(teacher_outputs, dim=-1)
    return -torch.mean(torch.sum(student_logits * teacher_logits, dim=-1) / temperature)

# Custom TrainingArguments class to include contrastive weight
class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, contrastive_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight

# Custom Trainer to include distillation and contrastive loss
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

        # Compute contrastive loss
        contrastive_loss_value = contrastive_loss(outputs_student.logits, avg_teacher_logits, self.args.temperature)

        # Combine student loss, distillation loss, and contrastive loss
        loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        total_loss = loss + self.contrastive_weight * contrastive_loss_value

        return (total_loss, outputs_student) if return_outputs else total_loss

# Define training arguments for full dataset and 6 epochs with checkpointing
training_args_full = DistillationTrainingArguments(
    output_dir=MODEL_OUTPUT,
    overwrite_output_dir=True,
    save_strategy="epoch",  # Save checkpoint after each epoch
    evaluation_strategy="epoch",  # Evaluate after each epoch
    num_train_epochs=6,  # Train for 6 epochs
    gradient_accumulation_steps=1,
    per_device_train_batch_size=BATCH_SIZE,
    save_total_limit=2,  # Limit the number of checkpoints to 2 (last and best)
    report_to="wandb",
    warmup_steps=200,
    lr_scheduler_type="cosine",
    learning_rate=LR,
    logging_steps=20,
    fp16=True,
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model="eval_loss",  # Use evaluation loss to choose the best model
    weight_decay=0.1,
    alpha=ALPHA,
    temperature=TEMPERATURE,
    contrastive_weight=CONTRASTIVE_WEIGHT,  # Use best-found contrastive weight
)

# Check if a checkpoint exists to resume training
resume_from_checkpoint = None
if (MODEL_OUTPUT / "pytorch_model.bin").exists():
    resume_from_checkpoint = str(MODEL_OUTPUT)

# Train the model using the full dataset
trainer_full = DistillationTrainer(
    student,
    training_args_full,
    teacher_models=teachers,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=full_eval_dataset,
)

# Train, resuming from the last checkpoint if available
trainer_full.train(resume_from_checkpoint=resume_from_checkpoint)