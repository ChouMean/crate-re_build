# CRATE-small Benchmark

This note documents the small benchmark added for the team run. The goal is to
verify that CRATE-small can be trained, fine-tuned, and evaluated on a limited
single-GPU setup.

## Hardware and Environment

- OS: Windows with PowerShell
- GPU: NVIDIA RTX 3060 6GB
- Python environment: local `.venv`
- Device: CUDA
- Precision: `float32`
- `torch.compile`: disabled for portability

Install dependencies:

```powershell
cd E:\Document\ML\CRATE-LM\crate-lm
.\.venv\Scripts\Activate.ps1
pip install torch tiktoken datasets tqdm requests
```

Check CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

## Model Configuration

The benchmark uses CRATE-small:

| Parameter | Value |
| --- | --- |
| Layers | 4 |
| Attention heads | 4 |
| Embedding dimension | 256 |
| Block size | 256 |
| Dropout | 0.1 |
| Parameters | about 14.46M |

For scratch training:

```text
batch_size = 8
block_size = 256
gradient_accumulation_steps = 4
tokens_per_iter = 8 * 256 * 4 = 8192
```

For fine-tuning:

```text
batch_size = 4
block_size = 256
gradient_accumulation_steps = 4
tokens_per_iter = 4 * 256 * 4 = 4096
learning_rate = 3e-5
```

## Datasets

| Dataset | Role | Approximate size |
| --- | --- | --- |
| Shakespeare | source/tested dataset | 300K tokens |
| Wikitext-2 | novel dataset 1 | 2M tokens |
| TinyStories | novel dataset 2 | 40M tokens |

The token counts come from the prepared datasets after tokenization. The
iteration counts are benchmark choices made to fit the small hardware setup.

## Files Added or Used

- `run_benchmark_small.ps1`: PowerShell runner for data preparation, scratch
  training, fine-tuning, and evaluation.
- `run_benchmark_small.sh`: Bash runner for the same small benchmark.
- `eval_checkpoint.py`: standalone checkpoint evaluation script reporting
  validation loss and perplexity.
- `config/train_shakespeare_small.py`: scratch training on Shakespeare.
- `config/train_wikitext2_small.py`: scratch training on Wikitext-2.
- `config/train_tinystories_small.py`: scratch training on TinyStories.
- `config/finetune_wikitext2_small.py`: fine-tune from Shakespeare to Wikitext-2.
- `config/finetune_tinystories_small.py`: fine-tune from Shakespeare to TinyStories.
- `benchmark_report_summary.txt`: text report with setup, commands, results,
  and interpretation.

Generated datasets, checkpoints, and output directories are intentionally not
committed.

## Reproduction Steps

Prepare data:

```powershell
python data/shakespeare/prepare.py
python data/wikitext2/prepare.py
python data/tinystories/prepare.py
```

Train from scratch:

```powershell
python train.py config/train_shakespeare_small.py --device=cuda
python train.py config/train_wikitext2_small.py --device=cuda
python train.py config/train_tinystories_small.py --device=cuda
```

Evaluate scratch checkpoints:

```powershell
python eval_checkpoint.py --out_dir=out-crate-small-wikitext2 --dataset=wikitext2 --device=cuda --eval_iters=50
python eval_checkpoint.py --out_dir=out-crate-small-tinystories --dataset=tinystories --device=cuda --eval_iters=50
```

Create fine-tuning checkpoints from Shakespeare:

```powershell
New-Item -ItemType Directory -Force out-crate-small-ft-wikitext2
Copy-Item out-crate-small-shakespeare\ckpt-400.pt out-crate-small-ft-wikitext2\ckpt.pt -Force

New-Item -ItemType Directory -Force out-crate-small-ft-tinystories
Copy-Item out-crate-small-shakespeare\ckpt-400.pt out-crate-small-ft-tinystories\ckpt.pt -Force
```

Fine-tune:

```powershell
python train.py config/finetune_wikitext2_small.py `
  --out_dir=out-crate-small-ft-wikitext2 `
  --device=cuda `
  --max_iters=1400 `
  --always_save_checkpoint=True

python train.py config/finetune_tinystories_small.py `
  --out_dir=out-crate-small-ft-tinystories `
  --device=cuda `
  --max_iters=1400 `
  --always_save_checkpoint=True
```

Evaluate fine-tuned checkpoints:

```powershell
python eval_checkpoint.py --out_dir=out-crate-small-ft-wikitext2 --dataset=wikitext2 --device=cuda --eval_iters=50
python eval_checkpoint.py --out_dir=out-crate-small-ft-tinystories --dataset=tinystories --device=cuda --eval_iters=50
```

## Current Results

| Dataset | Setup | Validation loss | Perplexity |
| --- | --- | ---: | ---: |
| Wikitext-2 | From scratch | 5.6646 | 288.48 |
| Wikitext-2 | Fine-tuned from Shakespeare | 6.7195 | 828.38 |
| TinyStories | From scratch | 2.2232 | 9.24 |
| TinyStories | Fine-tuned from Shakespeare | 4.2728 | 71.72 |

## Conclusion

CRATE-small was successfully trained and evaluated on the source dataset and two
novel datasets. In this small-scale setup, training from scratch on the target
dataset outperformed fine-tuning from the Shakespeare checkpoint on both
Wikitext-2 and TinyStories.

The likely explanation is that the Shakespeare source checkpoint is trained on a
small dataset and has a strong domain mismatch with both target datasets.
Therefore, these results should be interpreted as a limitation of this
particular source checkpoint and hardware-constrained setup, not as evidence
that fine-tuning is generally worse than scratch training.
