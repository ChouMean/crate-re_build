# Fine-tune a CRATE-small checkpoint on Wikitext-2 (novel dataset 1).
#
# Starts from the checkpoint produced by train_shakespeare_small.py and
# continues training on Wikitext-2 with a low learning rate.  Perplexity
# on the Wikitext-2 validation set is compared against the from-scratch
# result to measure transfer from the Shakespeare domain.
#
# Prerequisites:
#   1. python data/shakespeare/prepare.py
#   2. python data/wikitext2/prepare.py
#   3. python train.py config/train_shakespeare_small.py
#      (produces out-crate-small-shakespeare/ckpt.pt)
#
# Fine-tune:
#   python train.py config/finetune_wikitext2_small.py
#
# Evaluate:
#   python train.py config/eval_wikitext2_small.py

import time

out_dir = 'out-crate-small-ft-wikitext2'

wandb_log = False
wandb_project = 'crate-benchmark-small'
wandb_run_name = 'crate-small-ft-wikitext2-' + str(int(time.time()))

dataset = 'wikitext2'

# Only improve on validation loss (small dataset → easy to overfit)
always_save_checkpoint = False

# Must match the architecture of the pretrained checkpoint
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.1

# 4 micro-batch * 256 block * 4 grad-accum = 4 096 tokens/iter
batch_size = 4
block_size = 256
gradient_accumulation_steps = 4

max_iters = 1000
eval_interval = 100
eval_iters = 100
log_interval = 20

# Low constant LR for fine-tuning
learning_rate = 3e-5
decay_lr = False
weight_decay = 0.1

# Load the Shakespeare-trained checkpoint
init_from = 'resume'
ckpt_filename = 'ckpt.pt'   # set --out_dir=out-crate-small-shakespeare

dtype = 'float32'
compile = False
