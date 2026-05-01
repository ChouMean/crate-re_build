"""
Prepare the (Uncopyrighted) Pile dataset for language modeling.

Downloads monology/pile-uncopyrighted from HuggingFace, tokenizes with the
GPT-2 BPE tokenizer (tiktoken), and saves train.bin / val.bin.

This is the primary pre-training dataset used in the CRATE-LM paper.

Usage:
    python data/pile/prepare.py [--max_train_shards N]

The full Pile-uncopyrighted dataset is very large (~300 GB uncompressed).
Set MAX_TRAIN_SHARDS to limit how many streaming shards are consumed.
"""

import os
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

enc = tiktoken.get_encoding("gpt2")
MAX_TRAIN_SHARDS = 30  # Reduce for quick experiments; full dataset uses ~143 shards

def encode_and_write(iterable, filename, max_tokens=None):
    dtype = np.uint16
    # Write to a temporary growing list then flush to memmap
    all_ids = []
    for example in tqdm(iterable, desc=f'Processing {os.path.basename(filename)}'):
        text = example.get('text', '')
        if not text:
            continue
        ids = enc.encode_ordinary(text)
        ids.append(enc.eot_token)
        all_ids.extend(ids)
        if max_tokens and len(all_ids) >= max_tokens:
            all_ids = all_ids[:max_tokens]
            break

    arr = np.array(all_ids, dtype=dtype)
    m = np.memmap(filename, dtype=dtype, mode='w+', shape=(len(arr),))
    m[:] = arr
    m.flush()
    print(f"Wrote {filename}: {len(arr):,} tokens")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_train_shards', type=int, default=MAX_TRAIN_SHARDS,
                        help='Number of training shards to process (default: 30)')
    args = parser.parse_args()

    out_dir = os.path.dirname(__file__)

    # --- validation split ---
    print("Loading validation split...")
    val_dataset = load_dataset(
        "monology/pile-uncopyrighted",
        split="validation",
        streaming=True,
    )
    encode_and_write(
        val_dataset,
        os.path.join(out_dir, 'val.bin'),
        max_tokens=5_000_000,
    )

    # --- training split (streamed to avoid OOM) ---
    print(f"Loading training split (up to {args.max_train_shards} shards)...")
    train_dataset = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True,
    )
    encode_and_write(
        train_dataset,
        os.path.join(out_dir, 'train.bin'),
    )

    print("Done. Dataset saved to data/pile/")
