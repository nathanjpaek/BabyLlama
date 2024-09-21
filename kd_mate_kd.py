from transformers import (
    PreTrainedTokenizerFast,
    LlamaForCausalLM,
    LlamaConfig,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    ElectraForMaskedLM,
    get_cosine_schedule_with_warmup,
    AdamW
)

from typing import Dict, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Subset
from random import sample
from pathlib import Path
import wandb
from babylm_dataset import BabylmDataset
from torch.amp import autocast, GradScaler


#############
LR = 2.5e-4
BATCH_SIZE = 32
SEQ_LENGTH = 128

TEMPERATURE = 2.0
ALPHA = 0.5
#############

PATH = Path("./")

teacher_dir1 = PATH / './models/Llama-360M-G10'
teacher_dir2 = PATH / './models/GPT2-705M-orig'
generator_dir = PATH / './models/Electra-705M'

MODEL_NAME = f'Baby-Llama-58M-MATE-KD'
MODEL_OUTPUT = Path('./models') / MODEL_NAME
EVAL_SAMPLES = 8192

wandb_log = True

tokenizer_path = PATH / "models/gpt-clean-16000.json"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = "<pad>"
tokenizer.mask_token = "[MASK]"
tokenizer.unk_token= "[UNK]"

additional_special_tokens = {}
if tokenizer.mask_token not in tokenizer.get_vocab():
    additional_special_tokens['mask_token'] = "[MASK]"
if tokenizer.unk_token not in tokenizer.get_vocab():
    additional_special_tokens['unk_token'] = "[UNK]"
if additional_special_tokens:
    tokenizer.add_special_tokens(additional_special_tokens)
    print(f"Added special tokens: {list(additional_special_tokens.keys())}")

train_dataset = BabylmDataset(PATH / "Modified_data/babylm_10M_clean/gutenberg_10M_clean", SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True)
full_eval_dataset = BabylmDataset(PATH / "data/babylm_dev_clean", SEQ_LENGTH, tokenizer=tokenizer, offset=0)

eval_indices = sample(range(len(full_eval_dataset)), EVAL_SAMPLES)
eval_dataset = Subset(full_eval_dataset, eval_indices)

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
    max_position_embeddings=2*SEQ_LENGTH,
)

student = LlamaForCausalLM(config)
student.resize_token_embeddings(len(tokenizer))

teacher1 = LlamaForCausalLM.from_pretrained(teacher_dir1)
teacher1.resize_token_embeddings(len(tokenizer))

teacher2 = GPT2LMHeadModel.from_pretrained(teacher_dir2)
teacher2.resize_token_embeddings(len(tokenizer))

generator = ElectraForMaskedLM.from_pretrained(generator_dir)
generator.resize_token_embeddings(len(tokenizer))

teachers = [teacher1, teacher2]

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

print(f'model num parameters: student = {student.num_parameters()}')
print(f'model num parameters: teacher1 = {teacher1.num_parameters()}')
print(f'model num parameters: teacher2 = {teacher2.num_parameters()}')
print(f'model num parameters: generator = {generator.num_parameters()}')

generator_optimizer = AdamW(generator.parameters(), lr=LR)
generator_scheduler = get_cosine_schedule_with_warmup(
    optimizer=generator_optimizer,
    num_warmup_steps=200,
    num_training_steps=len(train_dataset) * 6 // BATCH_SIZE
)

class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature

def gumbel_softmax(logits, temperature):
    gumbel_noise = -torch.empty_like(logits).exponential_().log()
    y = (logits + gumbel_noise) / temperature
    return F.softmax(y, dim=-1)


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_models=None, generator_model=None, generator_optimizer=None, generator_scheduler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teachers = teacher_models
        self.generator = generator_model
        self.generator_optimizer = generator_optimizer
        self.generator_scheduler = generator_scheduler
        self.n_generator_iter = 10
        self.n_student_iter = 100
        self.n_repeat_batch = self.n_generator_iter + self.n_student_iter
        self.idx_pseudo = 0

        # Move teacher and generator models to the same device as the student
        for teacher in self.teachers:
            self._move_model_to_device(teacher, self.model.device)
            teacher.eval()
        self._move_model_to_device(self.generator, self.model.device)
        self.generator.eval()

        # FOR MIXED PRECISION:
        #self.scaler = GradScaler('cuda')

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs.get('labels')

        # with autocast('cuda'):
        # Generate perturbed inputs
        generator_outputs = self.generator(input_ids=input_ids, attention_mask=attention_mask)
        prediction_scores = generator_outputs.logits
        prediction_scores = F.gumbel_softmax(prediction_scores, tau=1.0, hard=True)

        # Create perturbed embeddings for teachers and student
        teacher_inps = []
        for teacher in self.teachers:
            teacher_inp = torch.matmul(prediction_scores, teacher.get_input_embeddings().weight)
            teacher_inps.append(teacher_inp)

        student_inp = torch.matmul(prediction_scores, model.get_input_embeddings().weight)

        # Get teacher and student logits
        teacher_logits = []
        for teacher, teacher_inp in zip(self.teachers, teacher_inps):
            with torch.no_grad():
                teacher_output = teacher(attention_mask=attention_mask, inputs_embeds=teacher_inp)
                teacher_logits.append(teacher_output.logits)

        avg_teacher_logits = torch.stack(teacher_logits).mean(dim=0)
        student_logits = model(attention_mask=attention_mask, inputs_embeds=student_inp).logits

        if self.idx_pseudo % self.n_repeat_batch < self.n_generator_iter:
            # Generator training phase
            loss = -F.kl_div(
                F.log_softmax(student_logits / self.args.temperature, dim=-1),
                F.softmax(avg_teacher_logits / self.args.temperature, dim=-1),
                reduction='batchmean'
            ) * (self.args.temperature ** 2)

            self.generator_optimizer.zero_grad()
            loss.backward()
            self.generator_optimizer.step()
            self.generator_scheduler.step()

            # FOR MIXED PRECISION:
            """self.generator_optimizer.zero_grad()  # Zero out gradients before backward pass
            scaled_loss = self.scaler.scale(loss)  # Scale the loss for mixed precision
            scaled_loss.backward()  # Perform the backward pass with scaled loss
            self.scaler.step(self.generator_optimizer)  # Step optimizer
            self.scaler.update()  # Update the scale factor
            self.generator_scheduler.step()"""

        else:
            # Student training phase
            original_student_logits = model(attention_mask=attention_mask, input_ids=input_ids).logits
            original_teacher_logits = []
            for teacher in self.teachers:
                with torch.no_grad():
                    teacher_output = teacher(attention_mask=attention_mask, input_ids=input_ids)
                    original_teacher_logits.append(teacher_output.logits)
            avg_original_teacher_logits = torch.stack(original_teacher_logits).mean(dim=0)

            loss_teach = F.kl_div(
                F.log_softmax(original_student_logits / self.args.temperature, dim=-1),
                F.softmax(avg_original_teacher_logits / self.args.temperature, dim=-1),
                reduction='batchmean'
            ) * (self.args.temperature ** 2)

            # loss_good = F.cross_entropy(original_student_logits, labels)
            loss_good = F.cross_entropy(original_student_logits.view(-1, original_student_logits.size(-1)), labels.view(-1))

            loss_adv = F.kl_div(
                F.log_softmax(student_logits / self.args.temperature, dim=-1),
                F.softmax(avg_teacher_logits / self.args.temperature, dim=-1),
                reduction='batchmean'
            ) * (self.args.temperature ** 2)

            loss = loss_good * (1/3) + (1/3) * loss_adv + (1/3) * loss_teach

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            # FOR MIXED PRECISION:
            """self.optimizer.zero_grad()  # Zero out gradients before backward pass
            scaled_loss = self.scaler.scale(loss)  # Scale the loss for mixed precision
            scaled_loss.backward()  # Perform the backward pass with scaled loss
            self.scaler.step(self.optimizer)  # Step optimizer
            self.scaler.update()  # Update the scale factor
            self.lr_scheduler.step()"""

        self.idx_pseudo += 1
        if self.idx_pseudo >= self.n_repeat_batch:
            self.idx_pseudo = 0

        return (loss, None) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        return loss.detach()

    def train(self, resume_from_checkpoint=None, **kwargs):
        return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

if wandb_log:
    wandb.login()
    wandb.init(project='babylm', name=MODEL_NAME)

training_args = DistillationTrainingArguments(
    output_dir=MODEL_OUTPUT,
    overwrite_output_dir=True,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    num_train_epochs=6,
    gradient_accumulation_steps=2,
    per_device_train_batch_size=BATCH_SIZE,
    save_total_limit=1,
    report_to="wandb",
    warmup_steps=200, 
    lr_scheduler_type="cosine",
    learning_rate=LR,
    logging_steps=20,
    fp16=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    weight_decay=0.1,
    alpha=ALPHA,
    temperature=TEMPERATURE,
)

trainer = DistillationTrainer(
    student,
    training_args,
    teacher_models=teachers,
    generator_model=generator,
    generator_optimizer=generator_optimizer,
    generator_scheduler=generator_scheduler,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

trainer.save_model(MODEL_OUTPUT)
tokenizer.save_pretrained(MODEL_OUTPUT)