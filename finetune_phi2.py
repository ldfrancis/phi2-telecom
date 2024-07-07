import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer, DataCollatorForLanguageModeling

from utils.constants import PHI2_MODEL_ID
from utils.data_utils import get_train_val_dataset, train_collate_fn
import sys
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset

import os

from peft import LoraConfig, get_peft_model


# model
model =  AutoModelForCausalLM.from_pretrained(
    PHI2_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(PHI2_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# dataset
def create_example_text(example):
    return {"text": example["question"] + " " + example["answer"]}

def tokenize_example(example):
    return tokenizer(
        example["question"],
        max_length=512, padding="max_length", truncation=True
    )
questions_train = load_dataset("json", data_files="data/questions.json")
questions: Dataset = questions_train["train"] # type: ignore
questions = questions.map(create_example_text)
dataset = questions.map(
    tokenize_example,
    batched=True,
    num_proc=4, # type: ignore
    remove_columns=questions.column_names # type: ignore
)
dataset = dataset.train_test_split(test_size=0.2, seed=45)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Train
model_dir = "data/ft"
os.makedirs(model_dir, exists_ok=True)
training_args = TrainingArguments(
    output_dir="data/ft",
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=20,
    per_device_train_batch_size=8, 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
)

trainer.train()
