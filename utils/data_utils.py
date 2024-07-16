import json
import math
import os
from typing import List, Dict
import pandas as pd

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BitsAndBytesConfig

from tqdm.auto import tqdm
import random



from utils.constants import (
    EMBED_MODEL_ID,
    PHI2_MODEL_ID, 
    TRAINING_ANSWER_FILE, 
    TRAINING_FILE, 
    TESTING_FILE
)

from utils.rag_utils import get_context


PROMPT = """Select the correct option for the question below

Question: 
{question}

Options:
{options}

Answer: """

PROMPT_wITH_CONTEXT = """Given the context in backticks

`{context}`

Select the correct option for the question below

Question: 
{question}

Options:
{options}

Answer: """


class MCQDataset(Dataset):
    """Dataset to be used for finetuning
    """
    def __init__(self, questions:List[Dict[str, str]], tokenizer) -> None:
        super().__init__()
        self.questions = questions
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.questions)
    
    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        question = self.questions[index]
        question_ids = self.tokenizer.encode(question["question"])
        answer_ids = self.tokenizer.encode(question["answer"])
        question_mask = [0]*len(question_ids)
        answer_mask = [1]*len(answer_ids)
        ids = question_ids+answer_ids
        input_ids = ids[:-1]
        output_ids = ids[1:]
        output_mask = (question_mask+answer_mask)[1:]
        return {
            "input_ids": torch.tensor(input_ids).long(),
            "output_ids": torch.tensor(output_ids).long(),
            "output_mask": torch.tensor(output_mask).float(),
            "padding_mask": torch.tensor([1]*len(output_mask)).float()
        }
    

class MCQTestDataset(MCQDataset):
    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        question = self.questions[index]
        question_ids = self.tokenizer.encode(question["question"])
        return {
            "input_ids": torch.tensor(question_ids).long(),
        }


class PHI2MCQDataset(MCQDataset):
    def __init__(self, questions: List[Dict[str, str]]) -> None:
        super().__init__(questions, AutoTokenizer.from_pretrained(PHI2_MODEL_ID))


class LMDataset(Dataset):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(PHI2_MODEL_ID)
        ctxs = []
        self.ctx_len = 1024+1
        files = os.listdir("data/doc/txt")
        for filename in files:
            with open("data/doc/txt"+filename, "r") as f:
                doc += f.read()
                tokens = self.tokenizer.encode(f.read())
            for i in range(math.ceil(len(tokens)/self.ctx_len)):
                s_idx = i*self.ctx_len
                e_idx = s_idx + self.ctx_len
                toks = tokens[s_idx:e_idx]
                if len(toks) < self.ctx_len:
                    diff = self.ctx_len - len(toks)
                    toks += [self.tokenizer.eos_token_id]*diff
                ctxs += [toks]
        self.ctxs = ctxs
        
    def __len__(self):
        return len(self.ctxs)

    def __getitem__(self, idx):
        ctx = self.ctxs[idx]
        inp = ctx[:-1]
        out = ctx[1:]
        return {
            "inp": torch.tensor(inp).long(),
            "out": torch.tensor(out).long(),
        }



def load_questions(file_path, split="train"):
    questions = []
    with open(file_path, "r") as f:
        for k,v in json.load(f).items():
            id_ = int(k.split(" ")[-1])
            options_list = [k_ for k_ in v.keys() if "option" in k_]
            options_ = [v[f"option {i}"] for i in range(1, len(options_list)+1)]
            # options_ = [f"{l}. "+v for v, l in zip(options_, ["A", "B", "C", "D", "E"])]
            v["id"] = id_
            v["options"] = options_
            if split=="train":
                v["correct_option_idx"] = int(v["answer"].split(":")[0].split(" ")[-1])-1
            questions.append(v)
        return questions


def clean_question(question):
    for num in [14, 15, 16, 17, 18]:
        question = question.replace(f"[3GPP Release {num}]", "")
    return question


def prompt_for_question(question):
    question = TRAIN_QUESTIONS[index]
    prompt = "The following are multiple choice questions (with answers) about Telecom documents.\n\n"
    # prompt = "Select the correct option for the question below.\n\n"
    seen_idxs = [index]
    for _ in range(0):
        idx = index
        while(idx in seen_idxs):
            idx = random.choice(range(len(TRAIN_QUESTIONS)))
        seen_idxs.append(idx)
        p_question = TRAIN_QUESTIONS[idx]
        prompt += f"{clean_question(p_question['question'])}\n"
        prompt += "\n".join(p_question["options"])
        prompt += f"\nAnswer: {p_question['answer_option']}\n\n"
    prompt += f"{clean_question(question['question'])}\n"
    prompt += "\n".join(question["options"])
    prompt += "\nAnswer: "
    return prompt

# correct_options = 0
# for i in range(len(TRAIN_QUESTIONS)):
#     correct_options += TRAIN_QUESTIONS[i]["answer_option"] == {1:"A",2:"B",3:"C",4:"D",5:"E"}[
#         model(**model.tokenizer(prompt_for_question(i), return_tensors="pt").to(model.device)).logits[:,-1,option_ids].squeeze().argmax().item()+1
#     ]
#     print(correct_options/(i+1))


def mcq_answer(prompt, model, tokenizer, option_ids):
    return {1:"A",2:"B",3:"C",4:"D",5:"E"}[
        model(**tokenizer(prompt, return_tensors="pt").to(model.device)).logits[:,-1,option_ids].squeeze().argmax().item()+1
    ]


def add_context_to_questions():
    if os.path.exists("data/train.json"):
        return
    num_val = int(0.2*len(TRAIN_QUESTIONS))
    # add context to train_questions
    progbar = tqdm(range(len(TRAIN_QUESTIONS)))
    progbar.desc = "Train contexts"
    for v in TRAIN_QUESTIONS:
        contexts = get_context([clean_question(v["question"])])
        v["contexts"] = contexts
        progbar.update(1)
        progbar.set_postfix({"id":v["id"]})
    progbar.close()

    progbar = tqdm(range(len(TEST_QUESTIONS)))
    progbar.desc = "Test contexts"
    for v in TEST_QUESTIONS:
        contexts = get_context([clean_question(v["question"])])
        v["contexts"] = contexts
        progbar.update(1)
        progbar.set_postfix({"id":v["id"]})
    progbar.close()
    
    progbar = tqdm(range(len(TEST_QUESTIONS_NEW)))
    progbar.desc = "New Test contexts"
    for v in TEST_QUESTIONS_NEW:
        contexts = get_context([clean_question(v["question"])])
        v["contexts"] = contexts
        progbar.update(1)
        progbar.set_postfix({"id":v["id"]})
    progbar.close()
    
    train_questions = TRAIN_QUESTIONS[:-num_val]
    val_questions = TRAIN_QUESTIONS[-num_val:]
    test_questions = TEST_QUESTIONS + TEST_QUESTIONS_NEW

    with open("data/train.json", "w") as f:
        json.dump(train_questions, f)
    with open("data/val.json", "w") as f:
        json.dump(val_questions, f)
    with open("data/test.json", "w") as f:
        json.dump(test_questions, f)

def create_prompt(question, train_questions, split="train"):
    prompt = "The following are multiple choice questions (with answers) about Telecom documents.\n\n"
    

def create_possible_train_mcqs():

    question = TRAIN_QUESTIONS[index]
    prompt = "The following are multiple choice questions (with answers) about Telecom documents.\n\n"
    seen_idxs = [index]
    for _ in range(1):
        idx = index
        while(idx in seen_idxs):
            idx = random.choice(range(len(TRAIN_QUESTIONS)))
        seen_idxs.append(idx)
        p_question = TRAIN_QUESTIONS[idx]
        prompt += f"{clean_question(p_question['question'])}\n"
        prompt += "\n".join(p_question["options"])
        prompt += f"\nAnswer: {p_question['answer_option']}\n\n"
    prompt += f"{clean_question(question['question'])}\n"
    prompt += "\n".join(question["options"])
    prompt += "\nAnswer: "
    return prompt


    train_mcqs = []
    with open("data/train.json", "w") as f:
        train_questions = json.load(f)

    for quest in train_questions:
        


def prepare_prompts():
    with open("data/train.json", "w") as f:
        train_questions = json.load(f)
    with open("data/val.json", "w") as f:
        val_questions = json.load(f)
    with open("data/test.json", "w") as f:
        test_questions = json.load(f)   

    # for v in train_questions:




def get_prompt_answer():
    post_fix = EMBED_MODEL_ID.split("/")[-1]
    if os.path.exists(f"data/train_questions_{post_fix}.json") and os.path.exists(f"data/test_questions_{post_fix}.json") :
        with open(f"data/train_questions_{post_fix}.json", "r") as f:
            train_questions = json.load(f)
        with open(f"data/test_questions_{post_fix}.json", "r") as f:
            test_questions = json.load(f)
        with open(f"data/val_questions_{post_fix}.json", "r") as f:
            val_questions = json.load(f)
        return train_questions, val_questions, test_questions
    

    
    with open(TRAINING_FILE, "r") as f:
        train_dict = json.loads(f.read())
    with open(TESTING_FILE, "r") as f:
        test_dict = json.loads(f.read())
    train_df = pd.read_csv(TRAINING_ANSWER_FILE)
    train_df = train_df.set_index("Question_ID")

    def get_questions(_dict, split="train"):
        questions = {}
        progbar = tqdm(range(len(_dict)))
        progbar.desc = split
        for key, value in _dict.items():
            quest = value["question"].replace("[3GPP Release 18]", "")\
                    .replace("[3GPP Release 17]", "").replace("[3GPP Release 16]", "")\
                    .replace("[3GPP Release 15]", "").replace("[3GPP Release 14]", ""),
            question = PROMPT.format(
                question=quest,
                options=(
                    "\n".join([
                        f"{k.split(' ')[-1]}) {v}" 
                        for k,v in value.items() if "option " == k[:7]
                    ])
                )
            )
            question_context = PROMPT_wITH_CONTEXT.format(
                context = get_context(quest),
                question=quest,
                options=(
                    "\n".join([
                        f"{k.split(' ')[-1]}) {v}" 
                        for k,v in value.items() if "option " == k[:7]
                    ])
                )
            )
            qid = int(key.split(' ')[-1])
            questions[qid] = {}
            questions[qid]["question"] = question
            questions[qid]["question_context"] = question_context 
            questions[qid]["id"] = qid
            if split == "train":
                questions[qid]["answer"] = (
                    f"The correct option is {train_df.loc[qid, 'Answer_ID']}) "+"".join(value["answer"]\
                                                                .split(":")[1:])
                )
                questions[qid]["answer_option"] = int(train_df.loc[qid, 'Answer_ID'])

            progbar.update(1)
        progbar.close()
            
        return questions

    train_questions = [v for v in get_questions(train_dict).values()]
    val_questions = train_questions[-int(0.2*len(train_questions)):]
    train_questions = train_questions[:-int(0.2*len(train_questions))]
    test_questions  = [v for v in get_questions(test_dict, "test").values()]

    with open(f"data/train_questions_{post_fix}.json", "w") as f:
        json.dump(train_questions, f)
    with open(f"data/val_questions_{post_fix}.json", "w") as f:
        json.dump(val_questions, f)
    with open(f"data/test_questions_{post_fix}.json", "w") as f:
        json.dump(test_questions, f)

    return train_questions, val_questions, test_questions
 

# def get_train_val_dataset():
#     train_questions, _ = get_prompt_answer()
#     train_questions = [v for v in train_questions.values()]
#     train_q, val_q = train_test_split(train_questions, test_size=0.2, random_state=32)
#     return PHI2MCQDataset(train_q), PHI2MCQDataset(val_q)


def train_collate_fn(batch):
    max_len = 0
    input_ids = []
    output_ids = []
    output_masks = []
    padding_masks = []
    for i in range(len(batch)):
        max_len = max(max_len, len(batch[i]["input_ids"]))
    for i in range(len(batch)):
        l = max_len - len(batch[i]["input_ids"])
        padding = torch.tensor([50256]*l)
        mask = torch.tensor([0]*l)
        input_ids.append(
            torch.cat((batch[i]["input_ids"], padding), dim=0)
        )
        output_ids.append(
            torch.cat((batch[i]["output_ids"], padding), dim=0)
        )
        output_masks.append(
            torch.cat((batch[i]["output_mask"], mask), dim=0)
        )
        padding_masks.append(
            torch.cat((batch[i]["padding_mask"], mask), dim=0)
        )

    input_ids = torch.stack(input_ids, dim=0).long()
    output_ids = torch.stack(output_ids, dim=0).long()
    output_masks = torch.stack(output_masks, dim=0)
    padding_masks = torch.stack(padding_masks, dim=0)

    return {
        "input_ids": input_ids,
        "output_ids": output_ids,
        "output_masks": output_masks,
        "padding_masks":padding_masks,
    }

TRAIN_QUESTIONS = load_questions("data/TeleQnA_training.txt")
TEST_QUESTIONS = load_questions("data/TeleQnA_testing1.txt", split="test")
TEST_QUESTIONS_NEW = load_questions("data/questions_new.txt", split="test")

if __name__=="__main__":
    add_context_to_questions()