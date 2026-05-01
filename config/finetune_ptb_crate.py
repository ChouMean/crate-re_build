# Fine-tune a pretrained CRATE model on Penn Treebank (novel fine-tune dataset 2).
#
# PTB is a classic Wall Street Journal corpus with ~1M training tokens.
# Fine-tuning here tests how a white-box CRATE model adapts to a structured,
# domain-specific text source.
#
# Prepare the data first:
#   python data/ptb/prepare.py
#
# Then fine-tune (place your pretrained checkpoint at out/ckpt.pt first):
#   python train.py config/finetune_ptb_crate.py
#
# After fine-tuning you can evaluate perplexity with:
#   python train.py config/eval_ptb_crate.py

import time

out_dir = 'out-crate-ft-ptb'

wandb_log = True
wandb_project = 'crate-benchmark'
wandb_run_name = 'crate-finetune-ptb-' + str(int(time.time()))

dataset = 'ptb'

# Save only when validation improves
always_save_checkpoint = False

# Match the architecture of the pretrained checkpoint
n_layer = 12
n_head = 12
n_embd = 768

# 4 micro-batch * 8 grad-accum * 512 block = 16,384 tokens/iter
batch_size = 4
block_size = 512
gradient_accumulation_steps = 8

max_iters = 1000         # PTB is small; converges quickly
eval_interval = 100
eval_iters = 100
log_interval = 10

# Low constant LR for fine-tuning
learning_rate = 3e-5
decay_lr = False
weight_decay = 0.1

# Start from a pretrained CRATE checkpoint
init_from = 'resume'

dtype = 'bfloat16'
compile = True
