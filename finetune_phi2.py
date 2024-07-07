import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer, DataCollatorForLanguageModeling

from utils.constants import PHI2_MODEL_ID
from utils.data_utils import get_train_val_dataset, train_collate_fn, get_prompt_answer
from utils.mcq_utils import evaluate_mcqs
import sys
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset

import json

import os

from peft import LoraConfig, get_peft_model


# Model
model =  AutoModelForCausalLM.from_pretrained(
    PHI2_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(PHI2_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# Dataset
def create_example_text(example):
    return {"text": example["question"] + " " + example["answer"]}

def tokenize_example(example):
    return tokenizer(
        example["question"],
        max_length=512, padding="max_length", truncation=True
    )

train_questions, test_questions = get_prompt_answer()
questions = [v for v in train_questions.values()]
with open("data/questions.json", "w") as f:
    json.dump(questions, f)
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
os.makedirs(model_dir, exist_ok=True)
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=512,  # reduce if running into out-of-memory issues
    lora_alpha=512,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'dense'],
    modules_to_save=["lm_head"],
    lora_dropout=0.05,
)
peft_model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir="data/ft",
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=20,
    num_train_epochs=3,
    per_device_train_batch_size=8, 
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
)

trainer.train()
trainer.model.save_pretrained("data/ft")
evaluate_mcqs(train_questions.values(), trainer.model, tokenizer)


breakpoint()