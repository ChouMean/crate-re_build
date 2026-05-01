"""
Prepare the Wikitext-2 dataset for language modeling.

Downloads wikitext-2-raw-v1 from HuggingFace, tokenizes with the GPT-2
BPE tokenizer (tiktoken) and saves train.bin / val.bin / test.bin.

Wikitext-2 contains ~2M training tokens and ~217K validation tokens.
It is much smaller than Wikitext-103 and well-suited for experiments on
limited hardware (novel dataset 1 in the small-model benchmark).

Usage:
    python data/wikitext2/prepare.py
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def process(example):
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

    out_dir = os.path.dirname(__file__)
    for split, dset in tokenized.items():
        out_split = 'val' if split == 'validation' else split
        arr_len = int(np.sum(dset['len'], dtype=np.int64))
        if arr_len == 0:
            print(f"Skipping empty split: {split}")
            continue
        filename = os.path.join(out_dir, f'{out_split}.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        idx = 0
        for example in tqdm(dset, desc=f'Writing {out_split}.bin'):
            ids = example['ids']
            if ids:
                chunk = np.array(ids, dtype=dtype)
                arr[idx: idx + len(chunk)] = chunk
                idx += len(chunk)
        arr.flush()
        print(f"Wrote {out_split}.bin: {arr_len:,} tokens")

    print("Done. Dataset saved to data/wikitext2/")
