"""
Prepare the Shakespeare dataset for fine-tuning / language-model evaluation.

Downloads the complete works of Shakespeare (word-level, not character-level),
tokenizes with the GPT-2 BPE tokenizer, and saves train.bin / val.bin.

This is one of the two novel fine-tuning datasets used in the CRATE-LM benchmark.

Usage:
    python data/shakespeare/prepare.py
"""

import os
import requests
import numpy as np
import tiktoken
from tqdm import tqdm

enc = tiktoken.get_encoding("gpt2")

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

if __name__ == '__main__':
    out_dir = os.path.dirname(__file__)

    # Download the text
    txt_path = os.path.join(out_dir, 'input.txt')
    if not os.path.exists(txt_path):
        print("Downloading Shakespeare text...")
        data = requests.get(SHAKESPEARE_URL).text
        with open(txt_path, 'w') as f:
            f.write(data)
    else:
        with open(txt_path, 'r') as f:
            data = f.read()

    print(f"Loaded {len(data):,} characters")

    # 90 / 10 train / val split
    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    for split_name, text in [('train', train_data), ('val', val_data)]:
        ids = enc.encode_ordinary(text)
        arr = np.array(ids, dtype=np.uint16)
        path = os.path.join(out_dir, f'{split_name}.bin')
        m = np.memmap(path, dtype=np.uint16, mode='w+', shape=(len(arr),))
        m[:] = arr
        m.flush()
        print(f"Wrote {path}: {len(arr):,} tokens")

    print("Done. Dataset saved to data/shakespeare/")
