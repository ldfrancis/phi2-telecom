import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils.rag_utils import create_keys
import numpy as np
import math
import os
from tqdm.auto import tqdm


embed_model_id = "BAAI/llm-embedder"
emb_model = AutoModel.from_pretrained(embed_model_id)
emb_tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
emb_model.to("cuda")

CHUNKS_DIR = "data/doc/chunks/"
CHUNKS_FILE = CHUNKS_DIR+"chunks.npy" 
chunks = np.load(CHUNKS_FILE)
embeds = []
bs = 8
steps = math.ceil(len(chunks)/bs)
progressbar = tqdm(range(steps))
for i in range(steps):
    start = i*bs
    end = (i+1)*bs
    batch = chunks[start:end]
    inp = emb_tokenizer(create_keys(chunks[start:end]), padding=True, truncation=True, return_tensors='pt')
    inp.to(emb_model.device)
    with torch.no_grad():
        _output = emb_model(**inp)
        _embed = _output.last_hidden_state[:, 0]
        _embed = torch.nn.functional.normalize(_embed, p=2, dim=1)
        embeds.append(_embed.cpu().numpy())
    progressbar.update(1)
progressbar.close()

# save embedding
embeds = np.concatenate(embeds, axis=0)
np.save(os.path.join(CHUNKS_DIR, "embeds.npy"), embeds)