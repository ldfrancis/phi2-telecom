import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer, DataCollatorForLanguageModeling

from utils.constants import PHI2_MODEL_ID, EMBED_MODEL_ID
from utils.data_utils import get_prompt_answer
from utils.mcq_utils import evaluate_mcqs
import sys
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset

import json

import os

from peft import LoraConfig, get_peft_model

USE_RAG = True
if USE_RAG:
    context_len = 1024
else:
    context_len = 512

# Model
model =  AutoModelForCausalLM.from_pretrained(
    PHI2_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(PHI2_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# Dataset
def create_example_text(example, rag=False):
    question_key = "question_context" if rag else "question"
    return {"text": example[question_key] + " " + example["answer"]}

def tokenize_example(example):
    return tokenizer(
        example["text"],
        max_length=context_len, padding="max_length", truncation=True
    )

train_questions, val_questions, test_questions = get_prompt_answer()

def get_data(fpath):
    data = load_dataset("json", data_files=fpath)
    return data["train"]


post_fix = EMBED_MODEL_ID.split("/")[-1]
questions_train = get_data(f"data/train_questions_{post_fix}.json").map(lambda x: create_example_text(x, rag=USE_RAG))
questions_val = get_data(f"data/val_questions_{post_fix}.json").map(lambda x: create_example_text(x, rag=USE_RAG))

dataset_train = questions_train.map(
    tokenize_example,
    batched=True,
    num_proc=4, # type: ignore
    remove_columns=questions_train.column_names # type: ignore
)
dataset_val = questions_val.map(
    tokenize_example,
    batched=True,
    num_proc=4, # type: ignore
    remove_columns=questions_train.column_names # type: ignore
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Train
model_dir = f"data/ft_{post_fix}"
os.makedirs(model_dir, exist_ok=True)
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=32,  # reduce if running into out-of-memory issues
    lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'dense'],
    modules_to_save=["lm_head"],
    lora_dropout=0.05,
)
peft_model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir=model_dir,
    eval_strategy="epoch",
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=10,
    num_train_epochs=6,
    per_device_train_batch_size=4, 
    fp16=True,
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    data_collator=data_collator,
)

trainer.train()
trainer.model.save_pretrained(model_dir)
evaluate_mcqs(val_questions, trainer.model, tokenizer, rag=True)


breakpoint()