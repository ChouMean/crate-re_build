# Train CRATE from scratch on OpenWebText (novel dataset 1).
#
# OpenWebText is a high-quality web-text corpus that replicates the WebText
# dataset used to train GPT-2.  Training here allows a direct comparison of
# CRATE vs GPT on the same pre-training data.
#
# Prepare the data first:
#   python data/openwebtext/prepare.py
#
# Then launch training (single GPU):
#   python train.py config/train_openwebtext.py
#
# Or with DDP (8 GPUs):
#   torchrun --standalone --nproc_per_node=8 train.py config/train_openwebtext.py

out_dir = 'out-crate-openwebtext'

wandb_log = True
wandb_project = 'crate-benchmark'
wandb_run_name = 'crate-openwebtext'

dataset = 'openwebtext'

# Model: CRATE-small  (matches 12L config from paper)
n_layer = 12
n_head = 12
n_embd = 768

# Batch: total ~0.5M tokens per step
# 12 micro-batch * 1024 block * 5 grad-accum * 8 GPUs = 491,520 tokens/step
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8  # 5 grad-accum-per-gpu * 8 GPUs; scale down for fewer GPUs if needed

# Training schedule (~300B tokens total)
max_iters = 600000
lr_decay_iters = 600000

# Evaluation
eval_interval = 1000
eval_iters = 200
log_interval = 10

# Optimizer
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
warmup_iters = 2000
min_lr = 6e-5
decay_lr = True

# Misc
init_from = 'scratch'
always_save_checkpoint = True
dtype = 'bfloat16'
compile = True
