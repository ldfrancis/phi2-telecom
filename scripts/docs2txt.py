""" Converts .docx files to text files.
"""
import docx
import os
import re
from tqdm.auto import tqdm

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

for filename in files:
    if filename.endswith(".docx"):
        progress_bar.set_postfix({"file": f"{DOCUMENT_DIR}/{filename}"})
        file_path = os.path.join(DOCUMENT_DIR, filename)
        doc = docx.Document(file_path)
        start_flag = False
        with open(os.path.join(TEXT_DIR, filename[:-4]+"txt"), "w") as f:
            for para in doc.paragraphs:
                text = para.text
                # start from the definitions section
                if re.search("[\d]+\tDefinitions", text) and not isInt(text[-1]):
                    start_flag = True
                if start_flag:
                    f.write(text)
            f.write("\n")
    progress_bar.update(1)
progress_bar.update(1)