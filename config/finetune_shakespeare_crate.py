# Fine-tune a pretrained CRATE model on Shakespeare (novel fine-tune dataset 1).
#
# This config fine-tunes a CRATE language model that was pre-trained (e.g. on
# the Pile) on the Shakespeare corpus.  The small dataset size means we train
# for very few iterations to avoid overfitting.
#
# Prepare the data first:
#   python data/shakespeare/prepare.py
#
# Then fine-tune (place your pretrained checkpoint at out/ckpt.pt first):
#   python train.py config/finetune_shakespeare_crate.py
#
# After fine-tuning you can evaluate perplexity with:
#   python train.py config/eval_shakespeare_crate.py

import time

out_dir = 'out-crate-ft-shakespeare'

wandb_log = True
wandb_project = 'crate-benchmark'
wandb_run_name = 'crate-finetune-shakespeare-' + str(int(time.time()))

dataset = 'shakespeare'

# Only save checkpoints when validation loss improves
always_save_checkpoint = False

# Match the architecture of the pretrained checkpoint
n_layer = 12
n_head = 12
n_embd = 768

# 1 micro-batch * 32 grad-accum * 1024 block = 32,768 tokens/iter
# Shakespeare has ~300K tokens, so 1 epoch ≈ 9 iters
batch_size = 1
block_size = 1024
gradient_accumulation_steps = 32

max_iters = 100          # fine-tune for a few epochs only
eval_interval = 10
eval_iters = 40
log_interval = 1

# Low constant LR for fine-tuning
learning_rate = 3e-5
decay_lr = False
weight_decay = 0.1

# Start from a pretrained CRATE checkpoint
init_from = 'resume'

dtype = 'bfloat16'
compile = True
