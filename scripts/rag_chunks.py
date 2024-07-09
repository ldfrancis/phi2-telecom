""" Converts .docx files to text files.
"""
import docx
import os
import re
import numpy as np
from tqdm.auto import tqdm

from transformers import AutoTokenizer
from utils.constants import PHI2_MODEL_ID

DOCUMENT_DIR = "data/doc/rel18"
TEXT_DIR = "data/doc/txt"
os.makedirs(TEXT_DIR, exist_ok=True)


def isInt(val):
    try:
        int(val)
        return True
    except ValueError:
        return False
    
files = os.listdir(DOCUMENT_DIR)
progress_bar = tqdm(range(len(files)))
chunks = []
sources = []
chunk_size = 1024

for filename in files:
    chunk = ""
    if filename.endswith(".docx"):
        progress_bar.set_postfix({"file": f"{DOCUMENT_DIR}/{filename}"})
        file_path = os.path.join(DOCUMENT_DIR, filename)
        doc = docx.Document(file_path)
        start_flag = False
        # with open(os.path.join(TEXT_DIR, filename[:-4]+"txt"), "w") as f:
        for para in doc.paragraphs:
            text = para.text
            if re.search("[\d]+\tDefinitions", text) and not isInt(text[-1]):
                start_flag = True
            if start_flag:
                if re.search("^\d[.]\d[.]?\d?\t[\w\d]+", text):
                    if len(chunk):
                        chunks += [chunk]
                        sources += [filename]
                    chunk = ""
                # f.write(text)
                # f.write("\n")
                if len(text) > 0:
                    chunk += text + "\n"
                    if len(chunk) > chunk_size:
                        chunks += [chunk]
                        sources += [filename]
                        chunk = ""
                # if len(chunk) >= 1024:
                #     chunks += [chunk]
                #     chunk = ""
    progress_bar.update(1)
progress_bar.update(1)

# save chunks
print("Creating chunks ...")
os.makedirs("data/doc/chunks", exist_ok=True)
np.save("data/doc/chunks/chunks.npy", np.array(chunks))
np.save("data/doc/chunks/sources.npy", np.array(sources))
