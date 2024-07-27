from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.constants import PHI2_MODEL_ID, EMBED_MODEL_ID
from utils.mcq_utils import evaluate_mcqs, answer_mcqs
from utils.data_utils import get_dataset
import pandas as pd
import json


post_fix = EMBED_MODEL_ID.split("/")[-1]

tokenizer = AutoTokenizer.from_pretrained(PHI2_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(f"/home/ubuntu/phi2-telecom/logs/model")
model.to("cuda")

train_questions, val_questions, test_questions = get_dataset()
testing1 = [int(k.split(" ")[-1]) for k, _ in json.load(open("data/TeleQnA_testing1.txt", "r")).items()]
mcqs = [m for m in test_questions if m["id"] in testing1]

answer_mcqs(mcqs, model, tokenizer, rag=True)
# test_questions = json.load(open("testq.json", "r"))
sample_submission = pd.read_csv("data/SampleSubmission.csv")
sample_submission["id"] = sample_submission["Question_ID"]
sample_submission = sample_submission.set_index("id")

for quest in mcqs:
    sample_submission.loc[quest["id"], "Answer_ID"] = int(quest["pred_option"]) 

sample_submission = sample_submission.reset_index(drop=True)

sample_submission.to_csv("submission.csv", index=False)