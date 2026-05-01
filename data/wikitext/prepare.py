"""
Prepare the Wikitext-103 dataset for language modeling.

Downloads wikitext-103-raw-v1 from HuggingFace, tokenizes with the GPT-2
BPE tokenizer (tiktoken) and saves train.bin / val.bin / test.bin.

Usage:
    python data/wikitext/prepare.py

Wikitext-103 contains 103M training tokens and ~218K validation tokens,
making it a standard benchmark for language modeling (perplexity).
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    def process(example):
        # Skip empty lines (Wikitext has many section headers / blank lines)
        text = example['text'].strip()
        if not text:
            return {'ids': [], 'len': 0}
        ids = enc.encode_ordinary(text)
        ids.append(enc.eot_token)
        return {'ids': ids, 'len': len(ids)}

    tokenized = dataset.map(
        process,
        remove_columns=['text'],
        desc="Tokenizing",
        num_proc=4,
    )

    for split, dset in tokenized.items():
        arr_len = int(np.sum(dset['len'], dtype=np.int64))
        if arr_len == 0:
            print(f"Skipping empty split: {split}")
            continue
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        idx = 0
        for example in tqdm(dset, desc=f'Writing {split}.bin'):
            ids = example['ids']
            if ids:
                chunk = np.array(ids, dtype=dtype)
                arr[idx: idx + len(chunk)] = chunk
                idx += len(chunk)
        arr.flush()
        print(f"Wrote {split}.bin: {arr_len:,} tokens")

    print("Done. Dataset saved to data/wikitext/")
