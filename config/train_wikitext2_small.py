# Train CRATE-small from scratch on Wikitext-2 (novel dataset 1).
#
# Wikitext-2 contains ~2 M training tokens extracted from Wikipedia and is
# a standard LM benchmark.  Training from scratch here lets us compare
# convergence and final perplexity against the fine-tuned variant.
#
# Prepare the data first:
#   python data/wikitext2/prepare.py
#
# Train:
#   python train.py config/train_wikitext2_small.py
#
# Evaluate:
#   python train.py config/eval_wikitext2_small.py

out_dir = 'out-crate-small-wikitext2'

wandb_log = False
wandb_project = 'crate-benchmark-small'
wandb_run_name = 'crate-small-wikitext2'

dataset = 'wikitext2'

# CRATE-small architecture
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.1

# 8 micro-batch * 256 block * 4 grad-accum = 8 192 tokens/iter
# Wikitext-2 has ~2 M tokens; 5 000 iters ≈ 20 epochs
batch_size = 8
block_size = 256
gradient_accumulation_steps = 4

max_iters = 5000
lr_decay_iters = 5000

# Evaluation
eval_interval = 500
eval_iters = 100
log_interval = 50

# Optimizer
learning_rate = 1e-3
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
warmup_iters = 200
min_lr = 1e-4
decay_lr = True

# Training setup
init_from = 'scratch'
always_save_checkpoint = True
dtype = 'float32'
compile = False
