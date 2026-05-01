# Train CRATE-small from scratch on Shakespeare (tested dataset).
#
# This is the base experiment: a compact CRATE model (4 layers, 4 heads,
# 256-dim embeddings) trained entirely on Shakespeare.  The resulting
# checkpoint is then used as the starting point for fine-tuning on
# Wikitext-2 and TinyStories.
#
# Prepare the data first:
#   python data/shakespeare/prepare.py
#
# Train:
#   python train.py config/train_shakespeare_small.py
#
# Evaluate:
#   python train.py config/eval_shakespeare_small.py

out_dir = 'out-crate-small-shakespeare'

wandb_log = False
wandb_project = 'crate-benchmark-small'
wandb_run_name = 'crate-small-shakespeare'

dataset = 'shakespeare'

# CRATE-small architecture (fits comfortably on a laptop GPU / CPU)
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.1

# Batch / block settings
# 8 micro-batch * 256 block * 4 grad-accum = 8 192 tokens/iter
batch_size = 8
block_size = 256
gradient_accumulation_steps = 4

# Shakespeare has ~300 K tokens; 2 000 iters ≈ 55 epochs
max_iters = 2000
lr_decay_iters = 2000

# Evaluation
eval_interval = 200
eval_iters = 100
log_interval = 20

# Optimizer
learning_rate = 1e-3
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
warmup_iters = 100
min_lr = 1e-4
decay_lr = True

# Training setup
init_from = 'scratch'
always_save_checkpoint = True
dtype = 'float32'   # safe default for CPU / older GPUs
compile = False     # disable torch.compile for portability
