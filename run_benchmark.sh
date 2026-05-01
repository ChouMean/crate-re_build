#!/usr/bin/env bash
# =============================================================================
# run_benchmark.sh  –  CRATE-LM Benchmark Suite
# =============================================================================
#
# Runs the full benchmark described in "Improving Neuron-level Interpretability
# with White-box Language Models" (Bai & Ma, CPAL 2025).
#
# Two benchmark tracks are implemented:
#   1. TRAIN FROM SCRATCH  –  pre-train CRATE on each dataset
#   2. FINE-TUNE            –  fine-tune a pre-trained CRATE checkpoint
#
# Datasets
# --------
#   Tested (paper) datasets :  Pile
#   Novel datasets (train)  :  OpenWebText, Wikitext-103
#   Novel datasets (ft/eval):  Shakespeare, Penn Treebank (PTB)
#
# Usage
# -----
#   # Full benchmark (all tracks)
#   bash run_benchmark.sh
#
#   # Only data preparation
#   bash run_benchmark.sh --prepare-only
#
#   # Only pre-training
#   bash run_benchmark.sh --train-only
#
#   # Only fine-tuning evaluation
#   bash run_benchmark.sh --finetune-only
#
#   # Override GPU ids (default: 0)
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_benchmark.sh
#
# Prerequisites
# -------------
#   pip install torch tiktoken datasets transformers tqdm requests
#   (optionally)  pip install wandb
#
# Notes
# -----
#   • All training commands default to single-GPU mode.  For multi-GPU DDP
#     replace `python train.py` with `torchrun --standalone --nproc_per_node=N train.py`
#     and set gradient_accumulation_steps accordingly.
#   • The Pile download is very large.  The prepare script limits to 30 shards
#     by default; pass --max_train_shards to change this.
#   • Fine-tuning requires a pretrained CRATE checkpoint at $PRETRAIN_CKPT_DIR.
#     Set the environment variable before running the fine-tune track.
# =============================================================================

set -euo pipefail

# ── Configurable paths ──────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PRETRAIN_CKPT_DIR="${PRETRAIN_CKPT_DIR:-out}"   # dir containing ckpt.pt for fine-tuning

# ── Flags ────────────────────────────────────────────────────────────────────
DO_PREPARE=1
DO_TRAIN=1
DO_FINETUNE=1

for arg in "$@"; do
    case "$arg" in
        --prepare-only)  DO_TRAIN=0;   DO_FINETUNE=0 ;;
        --train-only)    DO_PREPARE=0; DO_FINETUNE=0 ;;
        --finetune-only) DO_PREPARE=0; DO_TRAIN=0    ;;
    esac
done

export CUDA_VISIBLE_DEVICES

# ── Helpers ──────────────────────────────────────────────────────────────────
log() { echo ""; echo "==> $*"; echo ""; }

# ── 0. Sanity-check Python dependencies ─────────────────────────────────────
log "Checking Python dependencies"
python - <<'EOF'
import sys
missing = []
for pkg in ['torch', 'tiktoken', 'datasets', 'tqdm', 'requests']:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f"ERROR: missing packages: {', '.join(missing)}")
    print("Install with:  pip install " + " ".join(missing))
    sys.exit(1)
print("All dependencies present.")
EOF

# =============================================================================
# TRACK 1 – DATA PREPARATION
# =============================================================================
if [[ $DO_PREPARE -eq 1 ]]; then

    log "[PREPARE] Pile (tested dataset)"
    python data/pile/prepare.py

    log "[PREPARE] OpenWebText (novel training dataset 1)"
    python data/openwebtext/prepare.py

    log "[PREPARE] Wikitext-103 (novel training dataset 2)"
    python data/wikitext/prepare.py

    log "[PREPARE] LAMBADA (evaluation)"
    python data/lambada/prepare.py

    log "[PREPARE] Penn Treebank – PTB (novel fine-tune dataset 1)"
    python data/ptb/prepare.py

    log "[PREPARE] Shakespeare (novel fine-tune dataset 2)"
    python data/shakespeare/prepare.py

fi

# =============================================================================
# TRACK 2 – TRAINING FROM SCRATCH
# =============================================================================
if [[ $DO_TRAIN -eq 1 ]]; then

    # ---------- 2a. Tested dataset: Pile ------------------------------------
    log "[TRAIN] CRATE on Pile (tested dataset)"
    python train.py config/train_gpt2.py \
        --out_dir=out-crate-pile \
        --wandb_run_name=crate-pile \
        --compile=False

    log "[EVAL]  CRATE on Pile"
    python train.py config/eval_pile.py \
        --out_dir=out-crate-pile \
        --compile=False

    # ---------- 2b. Novel dataset 1: OpenWebText ----------------------------
    log "[TRAIN] CRATE on OpenWebText (novel dataset 1)"
    python train.py config/train_openwebtext.py \
        --compile=False

    log "[EVAL]  CRATE on OpenWebText"
    python train.py config/eval_openwebtext.py \
        --out_dir=out-crate-openwebtext \
        --compile=False

    # ---------- 2c. Novel dataset 2: Wikitext-103 ---------------------------
    log "[TRAIN] CRATE on Wikitext-103 (novel dataset 2)"
    python train.py config/train_wikitext103.py \
        --compile=False

    log "[EVAL]  CRATE on Wikitext-103"
    python train.py config/eval_wikitext103.py \
        --out_dir=out-crate-wikitext103 \
        --compile=False

    # ---------- 2d. Cross-dataset evaluation --------------------------------
    log "[EVAL]  CRATE (Pile-trained) on LAMBADA"
    python train.py config/eval_lambada.py \
        --out_dir=out-crate-pile \
        --compile=False

    log "[EVAL]  CRATE (Pile-trained) on Wikitext"
    python train.py config/eval_wikitext.py \
        --out_dir=out-crate-pile \
        --compile=False

fi

# =============================================================================
# TRACK 3 – FINE-TUNING EVALUATION
# =============================================================================
if [[ $DO_FINETUNE -eq 1 ]]; then

    if [[ ! -f "${PRETRAIN_CKPT_DIR}/ckpt.pt" ]]; then
        echo ""
        echo "WARNING: No pretrained checkpoint found at ${PRETRAIN_CKPT_DIR}/ckpt.pt"
        echo "         Skipping fine-tuning track."
        echo "         Set PRETRAIN_CKPT_DIR to your checkpoint directory and re-run"
        echo "         with --finetune-only, or run the full benchmark after training."
        echo ""
    else

        # ---------- 3a. Fine-tune on Shakespeare (novel ft dataset 1) -------
        log "[FINETUNE] CRATE on Shakespeare (novel fine-tune dataset 1)"
        python train.py config/finetune_shakespeare_crate.py \
            --out_dir="${PRETRAIN_CKPT_DIR}" \
            --compile=False

        log "[EVAL]     CRATE fine-tuned on Shakespeare"
        python train.py config/eval_shakespeare_crate.py \
            --out_dir=out-crate-ft-shakespeare \
            --compile=False

        # ---------- 3b. Fine-tune on PTB (novel ft dataset 2) ---------------
        log "[FINETUNE] CRATE on Penn Treebank (novel fine-tune dataset 2)"
        python train.py config/finetune_ptb_crate.py \
            --out_dir="${PRETRAIN_CKPT_DIR}" \
            --compile=False

        log "[EVAL]     CRATE fine-tuned on PTB"
        python train.py config/eval_ptb_crate.py \
            --out_dir=out-crate-ft-ptb \
            --compile=False

    fi

fi

log "Benchmark complete."
echo ""
echo "Summary of output directories:"
echo "  out-crate-pile/          – CRATE pre-trained on Pile"
echo "  out-crate-openwebtext/   – CRATE pre-trained on OpenWebText"
echo "  out-crate-wikitext103/   – CRATE pre-trained on Wikitext-103"
echo "  out-crate-ft-shakespeare/ – CRATE fine-tuned on Shakespeare"
echo "  out-crate-ft-ptb/        – CRATE fine-tuned on PTB"
echo ""
echo "To collect perplexity results, re-run the eval configs with --eval_only=True"
echo "on any of the above checkpoint directories."
