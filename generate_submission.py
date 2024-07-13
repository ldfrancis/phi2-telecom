from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import PHI2_MODEL_ID, EMBED_MODEL_ID
from utils.mcq_utils import evaluate_mcqs, answer_mcqs
from utils.data_utils import get_prompt_answer
import pandas as pd
import json


post_fix = EMBED_MODEL_ID.split("/")[-1]

tokenizer = AutoTokenizer.from_pretrained(PHI2_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(f"data/ft_{post_fix}")
model.to("cuda")

train_questions, val_questions, test_questions = get_prompt_answer()

answer_mcqs(test_questions, model, tokenizer, rag=True)
# test_questions = json.load(open("testq.json", "r"))
sample_submission = pd.read_csv("data/SampleSubmission.csv")
sample_submission["id"] = sample_submission["Question_ID"]
sample_submission = sample_submission.set_index("id")

for quest in test_questions:
    sample_submission.loc[quest["id"], "Answer_ID"] = int(quest["pred_option"]) 

sample_submission = sample_submission.reset_index(drop=True)

sample_submission.to_csv("submission.csv", index=False)