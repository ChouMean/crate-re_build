# Evaluate a CRATE model (fine-tuned or from-scratch) on Shakespeare.
#
# Usage:
#   python train.py config/eval_shakespeare_crate.py --out_dir=out-crate-ft-shakespeare

dataset = 'shakespeare'
eval_interval = 1000
eval_iters = 500
eval_only = True
wandb_log = False

n_layer = 12
n_head = 12
n_embd = 768

batch_size = 4
block_size = 1024
dtype = 'bfloat16'

init_from = 'resume'
