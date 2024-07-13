import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import math
from tqdm.auto import tqdm
from utils.constants import EMBED_MODEL_ID, EMBEDS_FILE, CHUNKS_FILE, EMBED_MODEL_TYPE


if EMBED_MODEL_TYPE == "HuggingFace":
    EMBED_MODEL = AutoModel.from_pretrained(EMBED_MODEL_ID)
    EMBED_TOKENIZER = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
    EMBED_MODEL.to("cuda")
else:
    EMBED_MODEL = SentenceTransformer(EMBED_MODEL_ID, trust_remote_code=True)
    EMBED_MODEL.max_seq_length = 8192

EMBEDS = None
CHUNKS = None

def create_queries(questions):
    prefix = "" if EMBED_MODEL_TYPE == "SentenceTransformer" else "Represent this query for retrieving relevant documents: "
    return [
        prefix + question
        for question in questions
    ]


def create_keys(texts):
    prefix = "" if EMBED_MODEL_TYPE == "SentenceTransformer" else "Represent this document for retrieval: "
    return [
        prefix + text
        for text in texts
    ]


def get_context(query, topk=1):
    global EMBEDS, CHUNKS
    if EMBEDS is None or CHUNKS is None:
        EMBEDS = np.load(EMBEDS_FILE)
        CHUNKS = np.load(CHUNKS_FILE)

    query = create_queries(query)
    if EMBED_MODEL_TYPE == "SentenceTransformer":
        q = torch.tensor(EMBED_MODEL.encode(query, prompt_name="query")).float().to(EMBED_MODEL.device)
    else:
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




# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer(EMBED_MODEL_ID, trust_remote_code=True)
# # In case you want to reduce the maximum length:
# model.max_seq_length = 8192

# queries = [
#     "how much protein should a female eat",
#     "summit define",
# ]
# documents = [
#     "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
#     "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
# ]

# query_embeddings = model.encode(queries, prompt_name="query")
# document_embeddings = model.encode(documents)

# scores = (query_embeddings @ document_embeddings.T) * 100
# print(scores.tolist())