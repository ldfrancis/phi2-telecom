from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import PHI2_MODEL_ID
from utils.mcq_utils import evaluate_mcqs
from utils.data_utils import get_prompt_answer
import json
import torch

# tq, val_questions, _ = get_prompt_answer()
with open("data/val_mcqs.json", "r") as f:
    val_questions = json.load(f)

# model = AutoModelForCausalLM.from_pretrained("/home/ubuntu/phi2-telecom/data/ft_gte-Qwen2-1.5B-instruct")
model =  AutoModelForCausalLM.from_pretrained(
    PHI2_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(PHI2_MODEL_ID)
model.to("cuda")

score, wrong_answers = evaluate_mcqs(val_questions, model, tokenizer, rag=True)

breakpoint()