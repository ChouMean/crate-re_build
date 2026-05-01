# Evaluate CRATE-small on Wikitext-2 (small-model benchmark).
#
# Usage (after from-scratch training):
#   python train.py config/eval_wikitext2_small.py --out_dir=out-crate-small-wikitext2
#
# Usage (after fine-tuning):
#   python train.py config/eval_wikitext2_small.py --out_dir=out-crate-small-ft-wikitext2

dataset = 'wikitext2'
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
