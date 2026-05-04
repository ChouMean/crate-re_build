import argparse
import math
import os
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from crate_overcomplete import CRATE, CRATEConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a CRATE checkpoint.")
    parser.add_argument("--out_dir", required=True, help="Directory containing ckpt.pt")
    parser.add_argument("--dataset", required=True, help="Dataset directory under data/")
    parser.add_argument("--ckpt_filename", default="ckpt.pt")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--progress_interval", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    device_type = "cuda" if "cuda" in args.device else "cpu"
    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available in this Python environment. "
            "Activate the same .venv used for training, or run with --device=cpu."
        )
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    ckpt_path = os.path.join(args.out_dir, args.ckpt_filename)
    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    checkpoint = torch.load(ckpt_path, map_location=args.device)

    model_args = checkpoint["model_args"]
    config = CRATEConfig(**model_args)
    model = CRATE(config)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    data_dir = os.path.join("data", args.dataset)
    data_path = os.path.join(data_dir, f"{args.split}.bin")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing split file: {data_path}")

    meta_path = os.path.join(data_dir, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        print(f"Dataset vocab_size from meta: {meta.get('vocab_size')}", flush=True)

    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    block_size = model_args["block_size"]
    if len(data) <= block_size:
        raise ValueError(f"{data_path} is too small for block_size={block_size}")

    def get_batch():
        ix = torch.randint(len(data) - block_size, (args.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
        if device_type == "cuda":
            x = x.pin_memory().to(args.device, non_blocking=True)
            y = y.pin_memory().to(args.device, non_blocking=True)
        else:
            x = x.to(args.device)
            y = y.to(args.device)
        return x, y

    losses = []
    print(
        f"Evaluating dataset={args.dataset}, split={args.split}, "
        f"eval_iters={args.eval_iters}, batch_size={args.batch_size}",
        flush=True,
    )
    with torch.no_grad():
        for i in range(args.eval_iters):
            x, y = get_batch()
            with ctx:
                _, loss = model(x, y)
            losses.append(float(loss.item()))
            if (i + 1) % args.progress_interval == 0 or i + 1 == args.eval_iters:
                mean_loss = sum(losses) / len(losses)
                print(f"{i + 1}/{args.eval_iters}: loss {mean_loss:.4f}", flush=True)

    val_loss = sum(losses) / len(losses)
    perplexity = math.exp(val_loss)
    print("")
    print(f"checkpoint_iter: {checkpoint.get('iter_num')}")
    print(f"checkpoint_best_val_loss: {checkpoint.get('best_val_loss')}")
    print(f"{args.dataset} {args.split} loss: {val_loss:.4f}")
    print(f"{args.dataset} {args.split} perplexity: {perplexity:.4f}")


if __name__ == "__main__":
    main()
