from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import PHI2_MODEL_ID
from utils.mcq_utils import evaluate_mcqs, evaluate_mcqs2
from utils.data_utils import get_dataset
import json
import torch

# tq, val_questions, _ = get_dataset()

model = AutoModelForCausalLM.from_pretrained("/home/ubuntu/phi2-telecom/logs/__model")
# model =  AutoModelForCausalLM.from_pretrained(
#     PHI2_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32
# )
tokenizer = AutoTokenizer.from_pretrained(PHI2_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model.to("cuda")

with open("/home/ubuntu/phi2-telecom/data/val-infgrad-stella_en_400M_v5.json", "r") as f:
    val_questions = json.load(f)

score, wrong_answers = evaluate_mcqs2(val_questions, model, tokenizer, rag=True)

breakpoint()