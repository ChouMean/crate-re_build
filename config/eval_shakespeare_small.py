# Evaluate CRATE-small on Shakespeare (small-model benchmark).
#
# Usage:
#   python train.py config/eval_shakespeare_small.py --out_dir=out-crate-small-shakespeare

dataset = 'shakespeare'
eval_interval = 1000
eval_iters = 200
eval_only = True
wandb_log = False

n_layer = 4
n_head = 4
n_embd = 256

batch_size = 8
block_size = 256
dtype = 'float32'

init_from = 'resume'
ckpt_filename = 'ckpt.pt'
compile = False
