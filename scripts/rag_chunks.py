import os
from click import progressbar
from tqdm.auto import tqdm
import numpy as np


TEXT_DIR = "data/doc/txt"
files = os.listdir(TEXT_DIR)

chunks = []
sources = []
chunk_size = 1024
chunk_overlap = 25
progressbar = tqdm(range(len(files)))

for filename in files:
    with open(os.path.join(TEXT_DIR, filename), "r") as f:
        text = f.read()
    for i in range(0, len(text), chunk_size-chunk_overlap):
        chunks.append(text[i:i+chunk_size])
        sources.append(filename)
    progressbar.update(1)
    progressbar.set_postfix({"file":filename})
progressbar.close()

# save chunks and sources
chunks_dir = "data/doc/chunks/"
os.makedirs(chunks_dir, exist_ok=True)
np.save(os.path.join(chunks_dir, "chunks.npy"), np.array(chunks))
np.save(os.path.join(chunks_dir, "sources.npy"), np.array(sources))
