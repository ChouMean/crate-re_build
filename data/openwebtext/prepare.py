"""
Prepare the OpenWebText dataset for language modeling.

Downloads the monology/openwebtext dataset from HuggingFace, tokenizes it
with the GPT-2 BPE tokenizer (via tiktoken), and saves train.bin / val.bin.

Usage:
    python data/openwebtext/prepare.py

The resulting files are ~17 GB for train and ~8.5 MB for val.
"""

import os
import multiprocessing
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# Number of workers for parallel tokenization
num_proc = max(1, multiprocessing.cpu_count() // 2)
num_proc_load_dataset = num_proc

# GPT-2 BPE tokenizer
enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)

    # Create a small held-out validation split (0.5%)
    split_dataset = dataset["train"].train_test_split(
        test_size=0.0005, seed=2357, shuffle=True
    )
    split_dataset['val'] = split_dataset.pop('test')

    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        return {'ids': ids, 'len': len(ids)}

    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="Tokenizing",
        num_proc=num_proc,
    )

    # Save each split to a binary file
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16  # GPT-2 vocab size is 50257, fits in uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = min(1024, len(dset))

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'Writing {split}.bin'):
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f"Wrote {split}.bin: {arr_len:,} tokens")

    print("Done. Dataset saved to data/openwebtext/")
