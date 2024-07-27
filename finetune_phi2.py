import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig, AutoTokenizer, DataCollatorForLanguageModeling

from utils.constants import PHI2_MODEL_ID, EMBED_MODEL_ID
from utils.data_utils import get_dataset, create_mcqs
from utils.mcq_utils import evaluate_mcqs
import sys
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
from peft import prepare_model_for_kbit_training
import bitsandbytes as bnb

import json

import os

from peft import LoraConfig, get_peft_model

USE_RAG = True
if USE_RAG:
    context_len = 768
else:
    context_len = 512

# Model
model =  AutoModelForCausalLM.from_pretrained(
    PHI2_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32#quantization_config=BitsAndBytesConfig(load_in_8bit=True)
)

# model = AutoModelForCausalLM.from_pretrained("/home/ubuntu/phi2-telecom/data/3gpp_train")
tokenizer = AutoTokenizer.from_pretrained(PHI2_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

def freeze_layers(model):
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad=True
            print(f"found lora param: {name}")
        elif "lm_head.modules_to_save" in name:
            param.requires_grad = True
            print(f"found lm_head param: {name}")
        else:
            param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Params: {trainable_params}")
    print(f"Total Params: {total_params}")

# Dataset
def create_example_text(example, rag=False):
    question_key = "question_context" if rag else "question"
    return {"text": example[question_key] + example["answer"]}

def tokenize_example(example):
    return tokenizer(
        example["text"],
        max_length=context_len, padding="max_length", truncation=True
    )

train_questions, val_questions, test_questions = get_dataset()

def get_data(fpath):
    data = load_dataset("json", data_files=fpath)
    return data["train"]


post_fix = EMBED_MODEL_ID.split("/")[-1]
questions_train = get_data(f"data/train-dataset.json").map(lambda x: create_example_text(x, rag=USE_RAG))
questions_val = get_data(f"data/val-dataset.json").map(lambda x: create_example_text(x, rag=USE_RAG))


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

# breakpoint()
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# tl = 0
# for val in dataset_train:
#     tl = max(tl, len(val["input_ids"]))
# for val in dataset_val:
#     tl = max(tl, len(val["input_ids"]))

# breakpoint()

# Train
model_dir = f"data/ft_{post_fix}"
os.makedirs(model_dir, exist_ok=True)
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=512,  # reduce if running into out-of-memory issues
    lora_alpha=512,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'dense'],
    modules_to_save=["lm_head"],
    lora_dropout=0.05,
)
model = get_peft_model(model, peft_config)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable Params: {trainable_params}")
print(f"Total Params: {total_params}")
# breakpoint()
# freeze_layers(model)

training_args = TrainingArguments(
    output_dir=model_dir,
    eval_strategy="steps",
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=20,
    eval_steps=100,
    num_train_epochs=8,
    per_device_train_batch_size=8, 
    save_strategy="no",  
    fp16=True,

)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    data_collator=data_collator,
)

trainer.train()
trainer.model.save_pretrained(model_dir)







# with open("data/val_mcqs.json", "r") as f:
#     val_mcqs = json.load(f)
evaluate_mcqs(val_questions, trainer.model, tokenizer, rag=True)


breakpoint()