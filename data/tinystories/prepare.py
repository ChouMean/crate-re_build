"""
Prepare the TinyStories dataset for language modeling.

Downloads roneneldan/TinyStories from HuggingFace, tokenizes with the
GPT-2 BPE tokenizer (tiktoken) and saves train.bin / val.bin.

TinyStories consists of short synthetic stories written at a child-reading
level, making it a distinctive domain for studying CRATE's generalisation
from the Shakespeare-trained small model (novel dataset 2 in the
small-model benchmark).

Only the first MAX_TRAIN_STORIES training stories are used by default to
keep data preparation fast on limited hardware.  Set MAX_TRAIN_STORIES=0
to use the full dataset (~2.5 M stories).

Usage:
    python data/tinystories/prepare.py
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

enc = tiktoken.get_encoding("gpt2")

# Limit training stories for speed on small machines (0 = use all)
MAX_TRAIN_STORIES = 200_000

if __name__ == '__main__':
    out_dir = os.path.dirname(__file__)

    dataset = load_dataset("roneneldan/TinyStories")

    for split in ['train', 'validation']:
        out_split = 'val' if split == 'validation' else 'train'
        dset = dataset[split]
        if split == 'train' and MAX_TRAIN_STORIES > 0:
            dset = dset.select(range(min(MAX_TRAIN_STORIES, len(dset))))
            print(f"Using first {len(dset):,} training stories")

        all_ids = []
        for example in tqdm(dset, desc=f'Tokenizing {out_split}'):
            text = example['text'].strip()
            if not text:
                continue
            ids = enc.encode_ordinary(text)
            ids.append(enc.eot_token)
            all_ids.extend(ids)

        arr = np.array(all_ids, dtype=np.uint16)
        path = os.path.join(out_dir, f'{out_split}.bin')
        m = np.memmap(path, dtype=np.uint16, mode='w+', shape=(len(arr),))
        m[:] = arr
        m.flush()
        print(f"Wrote {path}: {len(arr):,} tokens")

    print("Done. Dataset saved to data/tinystories/")
