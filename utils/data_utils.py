import json
import math
import os
from typing import List, Dict
import pandas as pd
import numpy as np
from utils.constants import CHUNKS_FILE
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BitsAndBytesConfig

from itertools import permutations

from tqdm.auto import tqdm
import random



from utils.constants import (
    EMBED_MODEL_ID,
    PHI2_MODEL_ID, 
    TRAINING_ANSWER_FILE, 
    TRAINING_FILE, 
    TESTING_FILE,
    USE_RAG_CONTEXT
)

from utils.rag_utils import get_chunks, get_embed_model


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


class MyDataLoader:
    def __init__(self, bs, tokenizer, topk):
        self.bs = bs
        self.tokenizer = tokenizer
        self.topk = topk
        self.init()

    def get_split(self):
        raise NotImplementedError()

    def init(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        post_fix = EMBED_MODEL_ID.replace("/", "-")
        with open(f"data/{self.get_split()}-{post_fix}.json", "r") as f:
            self.tdata = json.load(f)
        
        self.n_samples = math.ceil(len(self.tdata)/self.bs)
        self.idx = 0
        self.indices = [i for i in range(self.n_samples)]
        self.chunk_idxs = [i for i in range(4)]


    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        s_idx = idx * self.bs
        e_idx = min(len(self.tdata), s_idx + self.bs)
        batch = {"question_context":[], "answer":[]}
        for i in range(s_idx, e_idx):
            example = self.tdata[i]
            options = example["options"].copy()
            correct_option_idx = example["correct_option_idx"]
            correct_option_txt = options.pop(correct_option_idx)
            if self.get_split() != "val":
                random.shuffle(options)
                random.shuffle(self.chunk_idxs)
                new_correct_option_idx = random.randint(0, len(options))
            else:
                new_correct_option_idx = correct_option_idx
            options.insert(new_correct_option_idx, correct_option_txt)
            options_txt = "\n".join(
                [
                    f"{i+1}) {val[:-1] if val[-1] == '.' else val}" for i, val in enumerate(options)
                ]
            )
            context = "\n".join([example["chunks"][self.chunk_idxs[i]]  for i in range(self.topk)])
            # if self.get_split() != "val":
            #     idx = random.randint(1, len(context)-3)
            #     context = context[:idx] + correct_option_txt + context[idx:]
            prompt = prompt_w_context.format(
                choices=f"({','.join([f"{i+1}" for i in range(len(options))])})",
                question=clean_question(example["question"]),
                options=options_txt,
                context=context
            )
            question_context = f"{prompt}"
            answer =  f"{new_correct_option_idx+1}) {correct_option_txt} \nExplanation: {example["explanation"]}"
            batch["question_context"] += [question_context]
            batch["answer"] += [answer]

        self.tokenizer.padding_side = "left"
        q_tokens = self.tokenizer(batch["question_context"], padding="longest", return_tensors="pt")
        self.tokenizer.padding_side = "right"
        a_tokens = self.tokenizer(batch["answer"], padding="longest", return_tensors="pt")
        tokens = torch.cat([q_tokens["input_ids"], a_tokens["input_ids"]], dim=1)
        attn_masks = torch.cat([q_tokens["attention_mask"], a_tokens["attention_mask"]], dim=1)
        loss_mask = torch.cat([torch.zeros_like(q_tokens["attention_mask"]), a_tokens["attention_mask"]], dim=1)[:,1:]
        
        result = {
            "inp_ids":tokens[:,:-1],
            "inp_mask":attn_masks[:,:-1],
            "out_ids":tokens[:,1:],
            "out_mask":attn_masks[:,1:],
            "q_tokens": q_tokens,
            "a_tokens": a_tokens,

        }
        result["loss_mask"] = loss_mask * result["out_mask"]
        # result["out_ids"][:,:q_tokens["input_ids"].size(1)-10] = self.tokenizer.eos_token_id

        return result
        

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.n_samples:
            self.idx = 0
            raise StopIteration
        temp_idx = self.indices[self.idx]
        self.idx += 1
        return self[temp_idx]
        

class TrainDataLoader(MyDataLoader):
    def get_split(self):
        return "train"

    def __iter__(self):
        random.shuffle(self.indices)
        return super().__iter__()
    
class ValDataLoader(MyDataLoader):
    def get_split(self):
        return "val"


class TrainValDataLoader(MyDataLoader):
    def init(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        post_fix = EMBED_MODEL_ID.replace("/", "-")
        with open(f"data/train-{post_fix}.json", "r") as f:
            self.tdata = json.load(f)
        with open(f"data/val-{post_fix}.json", "r") as f:
            self.tdata += json.load(f)


def train_data_loader(bs, tokenizer, topk=4):
    tokenizer.pad_token = tokenizer.eos_token
    post_fix = EMBED_MODEL_ID.replace("/", "-")
    with open(f"data/train-{post_fix}.json", "r") as f:
        tdata = json.load(f)
    
    batch = {"question_context":[], "answer":[]}
    for example in tdata:
        options = example["options"].copy()
        correct_option_idx = example["correct_option_idx"]
        correct_option_txt = options.pop(correct_option_idx)
        random.shuffle(options)
        new_correct_option_idx = random.randint(0, len(options))
        options.insert(new_correct_option_idx, correct_option_txt)
        options_txt = "\n".join(
            [
                f"{i+1}) {val[:-1] if val[-1] == '.' else val}" for i, val in enumerate(options)
            ]
        )
        context = "\n".join(example["chunks"][:topk])
        prompt = prompt_w_context.format(
            choices=f"({','.join([f"{i+1}" for i in range(len(options))])})",
            question=clean_question(example["question"]),
            options=options_txt,
            context=context
        )
        question_context = f"{prompt}"
        answer =  f"{new_correct_option_idx+1}) {correct_option_txt}"
        batch["question_context"] += [question_context]
        batch["answer"] += [answer]
        
        if len(batch["answer"]) >= bs:
            tokenizer.padding_side = "left"
            q_tokens = tokenizer(batch["question_context"], padding="longest", return_tensors="pt")
            tokenizer.padding_side = "right"
            a_tokens = tokenizer(batch["answer"], padding="longest", return_tensors="pt")
            tokens = torch.cat([q_tokens["input_ids"], a_tokens["input_ids"]], dim=1)
            attn_masks = torch.cat([q_tokens["attention_mask"], a_tokens["attention_mask"]], dim=1)
            loss_mask = torch.cat([torch.zeros_like(q_tokens["attention_mask"]), a_tokens["attention_mask"]], dim=1)[:,1:]
            
            result = {
                "inp_ids":tokens[:,:-1],
                "inp_mask":attn_masks[:-1],
                "out_ids":tokens[:,1:],
                "out_mask":attn_masks[:,1:],
            }
            result["loss_mask"] = loss_mask * result["out_mask"]
            yield result

            batch["question_context"] = []
            batch["answer"] = []

    if len(batch["answer"]) > 0:
        tokenizer.padding_side = "left"
        q_tokens = tokenizer(batch["question_context"], padding="longest", return_tensors="pt")
        tokenizer.padding_side = "right"
        a_tokens = tokenizer(batch["answer"], padding="longest", return_tensors="pt")
        tokens = torch.cat([q_tokens["input_ids"], a_tokens["input_ids"]], dim=1)
        attn_masks = torch.cat([q_tokens["attention_mask"], a_tokens["attention_mask"]], dim=1)
        loss_mask = torch.cat([torch.zeros_like(q_tokens["attention_mask"]), a_tokens["attention_mask"]], dim=1)[:,1:]
        
        result = {
            "inp_ids":tokens[:,:-1],
            "inp_mask":attn_masks[:-1],
            "out_ids":tokens[:,1:],
            "out_mask":attn_masks[:,1:],
        }
        result["loss_mask"] = loss_mask * result["out_mask"]
        yield result




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
        # question = question.replace("3GPP", "")
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


def add_chunks_to_questions():
    with torch.inference_mode():
        TRAIN_QUESTIONS = load_questions("data/TeleQnA_training.txt")
        TEST_QUESTIONS = load_questions("data/TeleQnA_testing1.txt", split="test")
        TEST_QUESTIONS_NEW = load_questions("data/questions_new.txt", split="test")


        post_fix = EMBED_MODEL_ID.replace("/", "-")
        num_val = int(0.2*len(TRAIN_QUESTIONS))
        topk = 10
        # add context to train_questions
        progbar = tqdm(range(len(TRAIN_QUESTIONS)))
        progbar.desc = "Train chunks"
        for v in TRAIN_QUESTIONS:
            chunks = get_chunks([clean_question(v["question"])], topk)
            v["chunks"] = chunks
            progbar.update(1)
            progbar.set_postfix({"id":v["id"]})
        progbar.close()

        progbar = tqdm(range(len(TEST_QUESTIONS)))
        progbar.desc = "Test chunks"
        for v in TEST_QUESTIONS:
            chunks = get_chunks([clean_question(v["question"])], topk)
            v["chunks"] = chunks
            progbar.update(1)
            progbar.set_postfix({"id":v["id"]})
        progbar.close()
        
        progbar = tqdm(range(len(TEST_QUESTIONS_NEW)))
        progbar.desc = "New Test chunks"
        for v in TEST_QUESTIONS_NEW:
            chunks = get_chunks([clean_question(v["question"])], topk)
            v["chunks"] = chunks
            progbar.update(1)
            progbar.set_postfix({"id":v["id"]})
        progbar.close()
        
        train_questions = TRAIN_QUESTIONS[:-num_val]
        val_questions = TRAIN_QUESTIONS[-num_val:]
        test_questions = TEST_QUESTIONS + TEST_QUESTIONS_NEW

        with open(f"data/train-{post_fix}.json", "w") as f:
            json.dump(train_questions, f)
        with open(f"data/val-{post_fix}.json", "w") as f:
            json.dump(val_questions, f)
        with open(f"data/test-{post_fix}.json", "w") as f:
            json.dump(test_questions, f)


def obtain_top_sentences_from_chunks():
    post_fix = EMBED_MODEL_ID.replace("/", "-")
    EMBED_MODEL = get_embed_model()

    with open(f"data/train-{post_fix}.json", "r") as f:
        train_examples = json.load(f)
    with open(f"data/val-{post_fix}.json", "r") as f:
        val_examples = json.load(f)
    with open(f"data/test-{post_fix}.json", "r") as f:
        test_examples = json.load(f)

    progbar = tqdm(range(len(train_examples)))
    progbar.desc = "Train Sents"
    for example in train_examples:
        chunks = example["chunks"]
        sentences = []
        txt = ""
        for chunk in chunks:
            txt += chunk
            # temp_sentences = chunk.split(". ")
            # temp_sentences = [c for s in temp_sentences for c in s.split("\n") ]
            # temp_sentences = [s for s in temp_sentences if len(s) > 1]
            # temp_sentences = [s for s in temp_sentences if re.search("^\d[.]\d[.]?\d?\t[\w\d]+", s) is None]
            # sentences.extend(temp_sentences)

        # split text
        chunk_size = 256
        overlap = 25
        for i in range(0,len(txt),chunk_size-overlap):
            s_idx = i
            e_idx = s_idx + chunk_size
            sentences += [txt[s_idx:e_idx]]

        with torch.no_grad():
            k_embed = torch.tensor(EMBED_MODEL.encode(sentences)).float().to(EMBED_MODEL.device)

            prompt_name = "s2p_query" if "stella" in EMBED_MODEL_ID else "query"
            q_embed = torch.tensor(EMBED_MODEL.encode([clean_question(example["question"])], prompt_name=prompt_name)).float().to(EMBED_MODEL.device)
            if "stella" in EMBED_MODEL_ID:
                score = EMBED_MODEL.similarity(q_embed, k_embed)
            else:
                score = q_embed @ k_embed.T
            top_sentences = []
            for idx in score.argsort(dim=1).squeeze().tolist()[::-1]:
                top_sentences += [sentences[idx]]
            example["sentences"] = top_sentences
        # example["sentences"] = sentences
            
        progbar.update(1)
    progbar.close()

    progbar = tqdm(range(len(val_examples)))
    progbar.desc = "Val Sents"
    for example in val_examples:
        chunks = example["chunks"]
        txt = ""
        sentences = []
        for chunk in chunks:
            txt += chunk
            # temp_sentences = chunk.split(". ")
            # temp_sentences = [c for s in temp_sentences for c in s.split("\n") ]
            # temp_sentences = [s for s in temp_sentences if len(s) > 20]
            # sentences.extend(temp_sentences)


        # split text
        chunk_size = 256
        overlap = 25
        for i in range(0,len(txt),chunk_size-overlap):
            s_idx = i
            e_idx = s_idx + chunk_size
            sentences += [txt[s_idx:e_idx]]
        # sentences = txt[:2048]
        with torch.no_grad():
            k_embed = torch.tensor(EMBED_MODEL.encode(sentences)).float().to(EMBED_MODEL.device)

            prompt_name = "s2p_query" if "stella" in EMBED_MODEL_ID else "query"
            q_embed = torch.tensor(EMBED_MODEL.encode([clean_question(example["question"])], prompt_name=prompt_name)).float().to(EMBED_MODEL.device)
            if "stella" in EMBED_MODEL_ID:
                score = EMBED_MODEL.similarity(q_embed, k_embed)
            else:
                score = q_embed @ k_embed.T
            top_sentences = []
            for idx in score.argsort(dim=1).squeeze().tolist()[::-1]:
                top_sentences += [sentences[idx]]
            example["sentences"] = top_sentences#"".join(top_sentences)[:4096]
        # example["sentences"] = sentences

        progbar.update(1)
    progbar.close()

    progbar = tqdm(range(len(test_examples)))
    progbar.desc = "Test Sents"
    for example in test_examples:
        chunks = example["chunks"]
        sentences = []
        txt = ""
        for chunk in chunks:
            txt += chunk
            # temp_sentences = chunk.split(". ")
            # temp_sentences = [c for s in temp_sentences for c in s.split("\n") ]
            # temp_sentences = [s for s in temp_sentences if len(s) > 1]
            # sentences.extend(temp_sentences)

        # split text
        chunk_size = 256
        overlap = 25
        for i in range(0,len(txt),chunk_size-overlap):
            s_idx = i
            e_idx = s_idx + chunk_size
            sentences += [txt[s_idx:e_idx]]

        with torch.no_grad():
            k_embed = torch.tensor(EMBED_MODEL.encode(sentences)).float().to(EMBED_MODEL.device)

            prompt_name = "s2p_query" if "stella" in EMBED_MODEL_ID else "query"
            q_embed = torch.tensor(EMBED_MODEL.encode([clean_question(example["question"])], prompt_name=prompt_name)).float().to(EMBED_MODEL.device)
            if "stella" in EMBED_MODEL_ID:
                score = EMBED_MODEL.similarity(q_embed, k_embed)
            else:
                score = q_embed @ k_embed.T
            top_sentences = []
            for idx in score.argsort(dim=1).squeeze().tolist()[::-1]:
                top_sentences += [sentences[idx]]
            example["sentences"] = top_sentences
        # example["sentences"] = sentences
    
        progbar.update(1)
    progbar.close()

    with open(f"data/train-{post_fix}.json", "w") as f:
        json.dump(train_examples, f)
    with open(f"data/val-{post_fix}.json", "w") as f:
        json.dump(val_examples, f)
    with open(f"data/test-{post_fix}.json", "w") as f:
        json.dump(test_examples, f)


prompt_w_context = """Instruct: Given the context `{context}`, Answer the following question by selecting the most likely answer choice {choices} ensuring that the selected option can be found in the context

{question}

{options}
Output:"""


prompt_ = """Instruct: Answer the following question by selecting the most likely answer choice {choices}, please generate the answer choice

{question}

{options}
Output:"""

def get_dataset():
    post_fix = EMBED_MODEL_ID.replace("/", "-")
    with open(f"data/train-{post_fix}.json", "r") as f:
        train_examples = json.load(f)
    with open(f"data/val-{post_fix}.json", "r") as f:
        val_examples = json.load(f)
    with open(f"data/test-{post_fix}.json", "r") as f:
        test_examples = json.load(f)

    num_sentences = 10

    train_dataset = []
    progbar = tqdm(range(len(train_examples)))
    progbar.desc = "Train Data"
    for example in train_examples:
        # answer = example["answer"].split(":")[-1]
        # example["sentences"].insert(random.randint(0, len(example["sentences"])-1), answer)
        context = "".join(example["chunks"])[:4096]#example["answer"].split(":")[-1]#"".join(example["chunks"])#"\n".join(example["sentences"][:num_sentences])

        # question = (
        #     f"Instruct: Given the context `{context}`, "
        #     f"Select the correct option for the question, "
        #     f"\n{clean_question(example['question'])}\n"
        #     f"{'\n'.join([f'{i+1}. '+o for i,o in enumerate(example['options'])])}\nOutput: "
        # )
        question = prompt_w_context.format(
            choices="("+",".join([f"{k.split(' ')[-1]}" for k in example.keys() if "option " in k])+")",
            context=context,
            question=clean_question(example["question"]), 
            options="\n".join([f"{k.split(' ')[-1]}) "+ (v[:-1] if v[-1]=="." else v) for k,v in example.items() if "option " in k])
        )

        train_dataset += [
            {
                "id": example["id"],
                "question_context": question,
                "answer": (
                    f"{example["correct_option_idx"]+1})"
                    f" {example["options"][example["correct_option_idx"]]}"
                ),
                "explanation": example["explanation"],
                "answer_option": example["correct_option_idx"]+1
            }
        ]
        progbar.update(1)
    progbar.close()


    val_dataset = []
    progbar = tqdm(range(len(val_examples)))
    progbar.desc = "Train Data"
    for example in val_examples:
        context = "".join(example["chunks"])[:4096]#example["answer"].split(":")[-1]#"".join(example["chunks"])#"\n".join(example["sentences"][:num_sentences])

        # question = (
        #     f"Instruct: Given the context `{context}`, "
        #     f"Select the correct option for the question, "
        #     f"\n{clean_question(example['question'])}\n"
        #     f"{'\n'.join([f'{i+1}. '+o for i,o in enumerate(example['options'])])}\nOutput: "
        # )
        question = prompt_w_context.format(
            choices="("+",".join([f"{k.split(' ')[-1]}" for k in example.keys() if "option " in k])+")",
            context=context,
            question=clean_question(example["question"]), 
            options="\n".join([f"{k.split(' ')[-1]}) "+ (v[:-1] if v[-1]=="." else v) for k,v in example.items() if "option " in k])
        )
        val_dataset += [
            {   
                "id": example["id"],
                "question_context": question,
                "answer": (
                    f"{example["correct_option_idx"]+1})"
                    f" {example['options'][example['correct_option_idx']]}"
                ),
                "explanation": example["explanation"],
                "answer_option": example["correct_option_idx"]+1
            }
        ]
        progbar.update(1)
    progbar.close()

    test_dataset = []
    progbar = tqdm(range(len(test_examples)))
    progbar.desc = "Test Data"
    for example in test_examples:
        context = "".join(example["chunks"])[:6000]#"\n".join(example["sentences"][:num_sentences])

        # question = (
        #     f"Instruct: Given the context `{context}`, "
        #     f"Select the correct option for the question, "
        #     f"\n{clean_question(example['question'])}\n"
        #     f"{'\n'.join([f'{i+1}. '+o for i,o in enumerate(example['options'])])}\nOutput: "
        # )
        question = prompt_w_context.format(
            choices="("+",".join([f"{k.split(' ')[-1]}" for k,v in example.items() if ("option " in k and v is not None)])+")",
            context=context,
            question=clean_question(example["question"]), 
            options="\n".join([f"{k.split(' ')[-1]}) "+ (v[:-1] if v[-1]=="." else v) for k,v in example.items() if ("option " in k and v is not None)])
        )
        test_dataset += [
            {
                "id": example["id"],
                "question_context": question,
            }
        ]
        progbar.update(1)
    progbar.close()


    with open(f"data/train-dataset.json", "w") as f:
        json.dump(train_dataset, f)
    with open(f"data/val-dataset.json", "w") as f:
        json.dump(val_dataset, f)
    with open(f"data/test-dataset.json", "w") as f:
        json.dump(test_dataset, f)

    return train_dataset, val_dataset, test_dataset





        


def create_prompt(question, split="train"):
    prompt = "The following are multiple choice questions (with answers) about Telecom documents.\n\n"
    options = question["options"]
    context = "\n".join(question["contexts"][:2])

    def _prompt(q_text, options, correct_option, correct_id):
        idx = correct_id
        prompt = ""
        if USE_RAG_CONTEXT:
            prompt += "Knowing the information in backticks as a context:\n\n"
            prompt += f"`{context}`" + "\n\n"
        prompt += "The following are multiple choice questions (with answers) about Telecommunication documents.\n\n"
        prompt += clean_question(q_text) + "\n"
        _options = options.copy()
        # random.shuffle(_options)
        _options.insert(idx, correct_option)
        option_text = "\n".join([f"{o}. " + opt for o, opt in zip(["A","B","C","D","E"], _options)])
        option_answer = ["A","B","C","D","E"][idx]
        prompt += option_text
        prompt += "\nAnswer: "
        return {"prompt":prompt, "answer": f"{option_answer}. {correct_option}"}

    if split=="train":
        correct_id = question["correct_option_idx"]
        correct_option = options.pop(correct_id)
        prompts = []
        for idx in range(len(options)+1):
            for order in permutations(range(len(options)), len(options)):
                opt = [options[i] for i in order]
                prompt = _prompt(question["question"], opt, correct_option, idx)
                prompts += [prompt]
        return prompts[:1]
    elif split=="val":
        correct_id = question["correct_option_idx"]
        correct_option = options.pop(correct_id)
        return _prompt(question["question"], options, correct_option, correct_id)
    else:
        prompt = ""
        if USE_RAG_CONTEXT:
            prompt += "With the information below:\n\n"
            prompt += context + "\n\n"
        prompt += "The following are multiple choice questions (with answers) about Telecom documents.\n\n"
        prompt += clean_question(question["question"]) + "\n"
        options = [o for o in options if o is not None]
        option_text = "\n".join([f"{o}) " + opt for o, opt in zip(["A","B","C","D","E"], options)])
        prompt += option_text
        prompt += "\nAnswer: "
        return {"prompt": prompt}


def create_mcqs():
    train_mcqs = []
    with open("data/train.json", "r") as f:
        train_questions = json.load(f)

    for quest in train_questions:
        prompts = create_prompt(quest, split="train")
        for prompt in prompts:
            mcq = {**prompt, "id":quest["id"]}
            train_mcqs += [mcq]

    with open("data/train_mcqs.json", "w") as f:
        json.dump(train_mcqs, f)

    val_mcqs = []
    with open("data/val.json", "r") as f:
        val_questions = json.load(f)

    for quest in val_questions:
        prompt = create_prompt(quest, split="val")
        mcq = {**prompt, "id":quest["id"]}
        val_mcqs += [mcq]

    with open("data/val_mcqs.json", "w") as f:
        json.dump(val_mcqs, f)

    test_mcqs = []
    with open("data/test.json", "r") as f:
        test_questions = json.load(f)

    for quest in test_questions:
        prompt = create_prompt(quest, split="test")
        mcq = {**prompt, "id":quest["id"]}
        test_mcqs += [mcq]

    with open("data/test_mcqs.json", "w") as f:
        json.dump(test_mcqs, f)


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
    if os.path.exists(f"data/train_questions_{post_fix}.json"): 
        with open(f"data/train_questions_{post_fix}.json", "r") as f:
            train_questions = json.load(f)
    if  os.path.exists(f"data/test_questions_{post_fix}.json"):
        with open(f"data/test_questions_{post_fix}.json", "r") as f:
            test_questions = json.load(f)
    if  os.path.exists(f"data/val_questions_{post_fix}.json"):
        with open(f"data/val_questions_{post_fix}.json", "r") as f:
            val_questions = json.load(f)
        # return train_questions, val_questions, test_questions
    
    
    with open(TRAINING_FILE, "r") as f:
        train_dict = json.loads(f.read())
    with open(TESTING_FILE, "r") as f:
        test_dict = json.loads(f.read())
    with open("data/questions_new.txt", "r") as f:
        new_test_dict = json.loads(f.read())
    train_df = pd.read_csv(TRAINING_ANSWER_FILE)
    train_df = train_df.set_index("Question_ID")

    with open(f"data/train-{post_fix}.json", "w") as f:
        train_examples = json.load(f.read())

    train_examples = []
    test_examples = []
    for k,v in train_dict.items():
        qid = int(k.split(" ")[-1])
        v["id"] = qid
        train_examples += [v]
    for k,v in test_dict.items():
        qid = int(k.split(" ")[-1])
        v["id"] = qid
        test_examples += [v]
    for k,v in new_test_dict.items():
        qid = int(k.split(" ")[-1])
        v["id"] = qid
        test_examples += [v]

    num_val = int(0.2*len(train_examples))
    val_examples = train_examples[-num_val:]
    train_examples = train_examples[:-num_val]



    def get_questions(examples, split="train"):
        questions = []
        progbar = tqdm(range(len(examples)))
        progbar.desc = split
        for value in examples:
            qid = value["id"]
            options = [
                f"{v}" 
                for k,v in value.items() if ("option " == k[:7] and v is not None)
            ]

            if split in ["train","val"]:
                correct_option_num = int(train_df.loc[qid, 'Answer_ID'])
                correct_option_idx = correct_option_num-1
                correct_option_text = "".join(value["answer"].split(":")[1:])
                quest = clean_question(value["question"])
                context = get_context([quest])
               
                if False:#split == "train":
                    options.pop(correct_option_idx)
                    for idx in range(len(options)+1):
                        _options = options.copy()
                        _options.insert(idx, correct_option_text)
                        options_text = "".join([f"{i+1}) {opt}\n" for i,opt in enumerate(_options)])
                        answer_text = f"The correct option is {idx+1}) {correct_option_text}"
                        
                        question = PROMPT.format(
                            question=quest,
                            options=options_text,
                        )
                        question_context = PROMPT_wITH_CONTEXT.format(
                            context = context,
                            question=quest,
                            options=options_text
                        )
                        questions.append({
                            "question": question,
                            "question_context": question_context,
                            "answer": answer_text,
                            "answer_option": idx+1,
                            "id": f"{qid}_{idx}",
                        })
                else:
                    options_text = "".join([f"{i+1}) {opt}\n" for i,opt in enumerate(options)])
                    answer_text = f"The correct option is {int(correct_option_num)}) {correct_option_text}"
                    # quest = clean_question(value["question"])
                    question = PROMPT.format(
                        question=quest,
                        options=options_text,
                    )
                    question_context = PROMPT_wITH_CONTEXT.format(
                        context = context,
                        question=quest,
                        options=options_text
                    )
                    questions.append({
                        "question": question,
                        "question_context": question_context,
                        "answer": answer_text,
                        "answer_option": int(correct_option_num),
                        "id": f"{qid}",
                    })
            else:
                options_text = "".join([f"{i+1}) {opt}\n" for i,opt in enumerate(options)])
                quest = clean_question(value["question"])
                question = PROMPT.format(
                    question=quest,
                    options=options_text,
                )
                question_context = PROMPT_wITH_CONTEXT.format(
                    context=get_context([quest]),
                    question=quest,
                    options=options_text
                )
                questions.append({
                    "question": question,
                    "question_context": question_context,
                    "id": f"{qid}",
                })

            progbar.update(1)
            
        progbar.close()
            
        return questions

    
    if not os.path.exists(f"data/train_questions_{post_fix}.json"):
        train_questions = get_questions(train_examples, split="train")
        with open(f"data/train_questions_{post_fix}.json", "w") as f:
            json.dump(train_questions, f)
    if not os.path.exists(f"data/val_questions_{post_fix}.json"):
        val_questions = get_questions(val_examples, split="val")
        with open(f"data/val_questions_{post_fix}.json", "w") as f:
            json.dump(val_questions, f)
    if not os.path.exists(f"data/test_questions_{post_fix}.json"):
        test_questions = get_questions(test_examples, split="test")
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


def create_3gpp_data():
    chunks = np.load(CHUNKS_FILE)
    data = []
    for c in chunks:
        data += [{"text": str(c)}]

    with open("data/3gpp.json", "w") as f:
        json.dump(data, f)




def forward_pass(model, batch):
    inp_ids = batch["inp_ids"].to(model.device)
    attn_mask = batch["inp_mask"].to(model.device)
    result = model(input_ids=inp_ids, attention_mask=attn_mask)
    logits = result.logits
    return logits

def calc_loss(loss_fn, logits, batch):
    B, L, C = logits.shape
    target = batch["out_ids"].to(logits.device)
    mask = batch["loss_mask"].to(logits.device)
    loss = loss_fn(logits.reshape(-1, C), target.reshape(-1)) * mask.reshape(-1)
    loss = loss.sum()/mask.sum()
    return loss

def update(model, optimizer, loss_fn, batch, accumulate=True):
    logits = forward_pass(model, batch)
    loss = calc_loss(loss_fn, logits, batch)
    loss.backward()
    if not accumulate:
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()


if __name__=="__main__":
    add_chunks_to_questions()
    # obtain_top_sentences_from_chunks()
    # get_dataset()
    # create_mcqs()
    # create_3gpp_data()


    # from transformers import AutoModelForCausalLM
    # from peft import LoraConfig, get_peft_model
    # random.seed(1)
    # tokenizer = AutoTokenizer.from_pretrained(PHI2_MODEL_ID)
    # model = AutoModelForCausalLM.from_pretrained(PHI2_MODEL_ID, trust_remote_code=True)

    # peft_config = LoraConfig(
    #     task_type="CAUSAL_LM",
    #     r=32,  # reduce if running into out-of-memory issues
    #     lora_alpha=32,
    #     target_modules=['q_proj', 'k_proj', 'v_proj', 'dense'],
    #     modules_to_save=["lm_head"],
    #     lora_dropout=0.05,
    # )
    # model = get_peft_model(model, peft_config)

    # loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.eos_token_id, reduction='none')
    # optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
    # optimizer.zero_grad()
    # # gen = train_data_loader(16, tokenizer, topk=5)
    # # for batch in gen:
    # #     continue
    # # for batch in gen:
    # #     breakpoint()
    # #     continue
    # dloader = ValDataLoader(2, tokenizer, topk=5)
    # # dloader = iter(dloader_)
    # # breakpoint()
    # model.to("cuda")

    # model.train()
    # i = 0
    # progbar = tqdm(range(len(dloader)))
    # progbar.desc = "Train"
    # for batch in dloader:
    #     # with torch.inference_mode():
    #     #     model.eval()
    #     #     logits = forward_pass(model, batch)
    #     #     loss = calc_loss(loss_fn, logits, batch)
    #     #     update()
    #     #     breakpoint()
        
    #     loss = update(model, optimizer, loss_fn, batch, accumulate=False)
    #     tqdm.write(f"{loss}")
    #     torch.cuda.empty_cache()
    #     i+=1
    #     if i == 10:
    #         tks = model.generate(**tokenizer(["def init():"], return_tensors="pt").to(model.device), max_new_tokens=50)
    #         tqdm.write(tokenizer.decode(tks.squeeze().tolist()))
    #         breakpoint()
    #     progbar.update(1)
    # progbar.close()
    # breakpoint()
    # model.generate(tokenizer(["how far"], return_tensors="pt").to(model.device), max_new_tokens=10)

    print("DONE!")