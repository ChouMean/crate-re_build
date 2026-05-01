"""
Prepare the LAMBADA dataset for language-model evaluation.

LAMBADA tests a model's ability to predict the final word of a passage that
requires broader context.  We tokenize with the GPT-2 BPE tokenizer and
save val.bin for perplexity evaluation.

Usage:
    python data/lambada/prepare.py
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    out_dir = os.path.dirname(__file__)

    # Load openai_lambada (the standard split used in LM evaluations)
    dataset = load_dataset("EleutherAI/lambada_openai", trust_remote_code=True)

    for split_name, hf_split in [('val', 'test')]:
        dset = dataset[hf_split]
        all_ids = []
        for example in tqdm(dset, desc=f'Tokenizing {split_name}'):
            text = example['text']
            ids = enc.encode_ordinary(text)
            ids.append(enc.eot_token)
            all_ids.extend(ids)

        arr = np.array(all_ids, dtype=np.uint16)
        path = os.path.join(out_dir, f'{split_name}.bin')
        m = np.memmap(path, dtype=np.uint16, mode='w+', shape=(len(arr),))
        m[:] = arr
        m.flush()
        print(f"Wrote {path}: {len(arr):,} tokens")

    print("Done. Dataset saved to data/lambada/")
