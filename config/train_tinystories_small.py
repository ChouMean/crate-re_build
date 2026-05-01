# Train CRATE-small from scratch on TinyStories (novel dataset 2).
#
# TinyStories is a corpus of short synthetic children's stories.
# Its simple vocabulary and grammar make it a distinct domain from
# Shakespeare, and training from scratch here allows a direct comparison
# with the fine-tuned variant started from the Shakespeare checkpoint.
#
# Prepare the data first (downloads ~200 K stories by default):
#   python data/tinystories/prepare.py
#
# Train:
#   python train.py config/train_tinystories_small.py
#
# Evaluate:
#   python train.py config/eval_tinystories_small.py

out_dir = 'out-crate-small-tinystories'

wandb_log = False
wandb_project = 'crate-benchmark-small'
wandb_run_name = 'crate-small-tinystories'

dataset = 'tinystories'

# CRATE-small architecture
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.1

# 8 micro-batch * 256 block * 4 grad-accum = 8 192 tokens/iter
batch_size = 8
block_size = 256
gradient_accumulation_steps = 4

# 200 K stories ≈ 40 M tokens; 10 000 iters ≈ 2 epochs
max_iters = 10000
lr_decay_iters = 10000

# Evaluation
eval_interval = 1000
eval_iters = 200
log_interval = 100

# Optimizer
learning_rate = 1e-3
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
warmup_iters = 500
min_lr = 1e-4
decay_lr = True

# Training setup
init_from = 'scratch'
always_save_checkpoint = True
dtype = 'float32'
compile = False
