import json
from os import replace
from typing import List, Dict
import pandas as pd

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BitsAndBytesConfig


from utils.constants import (
    PHI2_MODEL_ID, 
    TRAINING_ANSWER_FILE, 
    TRAINING_FILE, 
    TESTING_FILE
)


PROMPT = """Select the correct option for the question below

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
        answer_ids = self.tokenizer.encode(question["question"])
        question_mask = [0]*len(question_ids)
        answer_mask = [1]*len(answer_ids)
        ids = question_ids+answer_ids
        input_ids = ids[:-1]
        output_ids = ids[1:]
        output_mask = (question_mask+answer_mask)[1:]
        return {
            "input_ids": torch.tensor(input_ids).long(),
            "output_ids": torch.tensor(output_ids).long(),
            "output_mask": torch.tensor(output_mask).float()
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


def get_prompt_answer():
    with open(TRAINING_FILE, "r") as f:
        train_dict = json.loads(f.read())
    with open(TESTING_FILE, "r") as f:
        test_dict = json.loads(f.read())
    train_df = pd.read_csv(TRAINING_ANSWER_FILE)
    train_df = train_df.set_index("Question_ID")

    def get_questions(_dict, split="train"):
        questions = {}
        for key, value in _dict.items():
            question = PROMPT.format(
                question=value["question"].replace("[3GPP Release 18]", "")\
                    .replace("[3GPP Release 17]", "").replace("[3GPP Release 16]", "")\
                    .replace("[3GPP Release 15]", "").replace("[3GPP Release 14]", ""),
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
            questions[qid]["id"] = qid
            if split == "train":
                questions[qid]["answer"] = (
                    f"{train_df.loc[qid, 'Answer_ID']}):"+"".join(value["answer"]\
                                                                .split(":")[1:])
                )
                questions[qid]["answer_option"] = train_df.loc[qid, 'Answer_ID']
            
        return questions

    train_questions = get_questions(train_dict)
    test_questions  = get_questions(test_dict, "test")

    return train_questions, test_questions
 

def get_train_val_dataset():
    train_questions, _ = get_prompt_answer()
    train_questions = [v for v in train_questions.values()]
    train_q, val_q = train_test_split(train_questions, test_size=0.2, random_state=32)
    return PHI2MCQDataset(train_q), PHI2MCQDataset(val_q)


def train_collate_fn(batch):
    max_len = 0
    input_ids = []
    output_ids = []
    output_masks = []
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

    input_ids = torch.stack(input_ids, dim=0).long()
    output_ids = torch.stack(output_ids, dim=0).long()
    output_masks = torch.stack(output_masks, dim=0)

    return {
        "input_ids": input_ids,
        "output_ids": output_ids,
        "output_masks": output_masks,
    }


if __name__=="__main__":
    print(get_train_val_dataset()[0][0])