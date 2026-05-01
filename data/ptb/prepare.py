"""
Prepare the Penn Treebank (PTB) dataset for language modeling.

Downloads ptb-text-only from HuggingFace, tokenizes with the GPT-2 BPE
tokenizer (tiktoken), and saves train.bin / val.bin / test.bin.

PTB is a classic benchmark with ~1M training tokens derived from Wall Street
Journal articles.

Usage:
    python data/ptb/prepare.py
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    out_dir = os.path.dirname(__file__)

    dataset = load_dataset("ptb-text-only/ptb_text_only", trust_remote_code=True)

    for split in ['train', 'validation', 'test']:
        out_split = 'val' if split == 'validation' else split
        dset = dataset[split]
        all_ids = []
        for example in tqdm(dset, desc=f'Tokenizing {split}'):
            text = example['sentence']
            ids = enc.encode_ordinary(text)
            ids.append(enc.eot_token)
            all_ids.extend(ids)

        arr = np.array(all_ids, dtype=np.uint16)
        path = os.path.join(out_dir, f'{out_split}.bin')
        m = np.memmap(path, dtype=np.uint16, mode='w+', shape=(len(arr),))
        m[:] = arr
        m.flush()
        print(f"Wrote {path}: {len(arr):,} tokens")

    print("Done. Dataset saved to data/ptb/")
