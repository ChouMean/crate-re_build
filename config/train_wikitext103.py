# Train CRATE from scratch on Wikitext-103 (novel dataset 2).
#
# Wikitext-103 is a curated Wikipedia-based corpus widely used as a language-
# model benchmark.  Training CRATE from scratch here lets us evaluate
# convergence and perplexity on a clean, factual text domain.
#
# Prepare the data first:
#   python data/wikitext/prepare.py
#
# Then launch training (single GPU, suitable for small-scale experiment):
#   python train.py config/train_wikitext103.py
#
# Or with DDP:
#   torchrun --standalone --nproc_per_node=4 train.py config/train_wikitext103.py

out_dir = 'out-crate-wikitext103'

wandb_log = True
wandb_project = 'crate-benchmark'
wandb_run_name = 'crate-wikitext103'

dataset = 'wikitext'   # points to data/wikitext/

# Model: CRATE-small
n_layer = 12
n_head = 12
n_embd = 768

# Wikitext-103 is ~103M training tokens — use a smaller batch and fewer steps
# than the full Pile run so training completes in a reasonable time.
# 8 micro-batch * 512 block * 4 grad-accum * 4 GPUs = 65,536 tokens/step
batch_size = 8
block_size = 512
gradient_accumulation_steps = 4 * 4  # 4 grad-accum-per-gpu * 4 GPUs

# ~50B tokens total (adequate for this corpus size)
max_iters = 100000
lr_decay_iters = 100000

# Evaluation
eval_interval = 500
eval_iters = 200
log_interval = 10

# Optimizer
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
warmup_iters = 1000
min_lr = 6e-5
decay_lr = True

# Misc
init_from = 'scratch'
always_save_checkpoint = True
dtype = 'bfloat16'
compile = True
