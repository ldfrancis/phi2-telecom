import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import math
from tqdm.auto import tqdm
from utils.constants import EMBED_MODEL_ID, EMBEDS_FILE, CHUNKS_FILE

EMBED_MODEL = AutoModel.from_pretrained(EMBED_MODEL_ID)
EMBED_TOKENIZER = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
EMBEDS = None
CHUNKS = None

def create_queries(questions):
    return [

    "Represent this query for retrieving relevant documents: "+ question
    for question in questions
    ]


def create_keys(texts):
    return ["Represent this document for retrieval: " + text
        for text in texts
    ]


def get_context(query, topk=2):
    global EMBEDS, CHUNKS
    if EMBEDS is None:
        EMBEDS = np.load(EMBEDS_FILE)
        CHUNKS = np.load(CHUNKS_FILE)

    query = create_queries(query)
    q = EMBED_TOKENIZER(query, padding=True, truncation=True, return_tensors='pt')
    q.to(EMBED_MODEL.device)
    q = EMBED_MODEL(**q).last_hidden_state[:, 0]
    q = torch.nn.functional.normalize(q, p=2, dim=1)

    bs = 256
    scores = []
    steps = math.ceil(EMBEDS.shape[0]/bs)
    for i in range(steps):
        start = i*bs
        end = (i+1)*bs
        k = torch.tensor(EMBEDS[start:end]).float().to(EMBED_MODEL.device)
        with torch.no_grad():
            score = q @ k.T
            scores.append(score.cpu().numpy())

    scores = np.concatenate(scores, axis=1)
    args = np.argsort(scores, axis=1)[:,::-1][:,:topk]

    context = ""
    for i in range(len(query)):
        for j in range(topk):
            context += CHUNKS[args[i,j]]
            context += "\n"
        context += "\n"

    return context