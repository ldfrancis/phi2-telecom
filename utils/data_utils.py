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


def get_prompt_answer():

    if os.path.exists(f"data/train_questions_{EMBED_MODEL_ID}.json") and os.path.exists(f"data/test_questions_{EMBED_MODEL_ID}.json") :
        with open(f"data/train_questions_{EMBED_MODEL_ID}.json", "r") as f:
            train_questions = json.load(f)
        with open(f"data/test_questions_{EMBED_MODEL_ID}.json", "r") as f:
            test_questions = json.load(f)
        with open(f"data/val_questions_{EMBED_MODEL_ID}.json", "r") as f:
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

    with open(f"data/train_questions_{EMBED_MODEL_ID}.json", "w") as f:
        json.dump(train_questions, f)
    with open(f"data/val_questions_{EMBED_MODEL_ID}.json", "w") as f:
        json.dump(val_questions, f)
    with open(f"data/test_questions_{EMBED_MODEL_ID}.json", "w") as f:
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


if __name__=="__main__":
    print(get_train_val_dataset()[0][0])