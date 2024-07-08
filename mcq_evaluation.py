from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import PHI2_MODEL_ID
from utils.mcq_utils import evaluate_mcqs
from utils.data_utils import get_prompt_answer
import torch

train_questions, val_questions, test_questions = get_prompt_answer()
model = AutoModelForCausalLM.from_pretrained("data/ft")
# model =  AutoModelForCausalLM.from_pretrained(
#     PHI2_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32
# )
tokenizer = AutoTokenizer.from_pretrained(PHI2_MODEL_ID)
model.to("cuda")

evaluate_mcqs(val_questions, model, tokenizer, rag=True)

breakpoint()