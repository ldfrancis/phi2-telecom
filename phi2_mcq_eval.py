import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.constants import PHI2_MODEL_ID
from utils.data_utils import get_prompt_answer, get_train_val_dataset, train_collate_fn
import sys
from tqdm.auto import tqdm

from peft import LoraConfig, get_peft_model


model =  AutoModelForCausalLM.from_pretrained(
    PHI2_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(PHI2_MODEL_ID)

device = "cuda:0"
model.to(device)

train_questions, test_questions = get_prompt_answer()


def train_eval():
    correct_count = 0
    total_count = 0
    for i, (k,v) in enumerate(train_questions.items()):
        question = v["question"]
        question_toks = tokenizer(question, return_tensors="pt")
        question_toks.to(model.device)
        result_toks = model.generate(**question_toks, max_new_tokens=40).squeeze().tolist()
        print(tokenizer.decode(result_toks))
        print("target: ", v["answer"], v["answer_option"])

train_eval()



