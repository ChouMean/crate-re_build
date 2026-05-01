# Evaluate a CRATE model trained on Wikitext-103.
#
# Usage:
#   python train.py config/eval_wikitext103.py --out_dir=out-crate-wikitext103

dataset = 'wikitext'
eval_interval = 3000
eval_iters = 1000
eval_only = True
wandb_log = False

n_layer = 12
n_head = 12
n_embd = 768

batch_size = 4
block_size = 512
dtype = 'bfloat16'

init_from = 'resume'
