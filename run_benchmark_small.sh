#!/usr/bin/env bash
# =============================================================================
# run_benchmark_small.sh  –  Small-model CRATE benchmark (laptop / single GPU)
# =============================================================================
#
# Runs the lightweight benchmark using CRATE-small (4 layers, 4 heads, 256-dim)
# designed to complete on a single GPU or even a CPU in a reasonable time.
#
# Two benchmark tracks:
#   1. TRAIN FROM SCRATCH  –  Shakespeare (tested dataset) + Wikitext-2 and
#                             TinyStories (novel datasets)
#   2. FINE-TUNE            –  fine-tune the Shakespeare checkpoint on the two
#                             novel datasets and evaluate perplexity
#
# Datasets
# --------
#   Tested dataset  :  Shakespeare (~1 MB, ~300 K tokens)
#   Novel dataset 1 :  Wikitext-2  (~2 M tokens from Wikipedia)
#   Novel dataset 2 :  TinyStories (~40 M tokens of synthetic children's stories)
#
# Usage
# -----
#   # Full benchmark (all tracks)
#   bash run_benchmark_small.sh
#
#   # Only data preparation
#   bash run_benchmark_small.sh --prepare-only
#
#   # Only training from scratch
#   bash run_benchmark_small.sh --train-only
#
#   # Only fine-tuning (requires scratch-trained Shakespeare checkpoint)
#   bash run_benchmark_small.sh --finetune-only
#
#   # Run on CPU only (slow but works anywhere)
#   DEVICE=cpu bash run_benchmark_small.sh
#
# Prerequisites
# -------------
#   pip install torch tiktoken datasets tqdm requests
#
# Notes
# -----
#   • All commands use compile=False for portability; add --compile=True if
#     you have PyTorch 2.0+ and want extra speed.
#   • DEVICE defaults to 'cuda' if a GPU is present, else 'cpu'.
#   • The Shakespeare checkpoint is saved to out-crate-small-shakespeare/ckpt.pt
#     and is reused for both fine-tuning steps.
# =============================================================================

set -euo pipefail

# ── Auto-detect device ───────────────────────────────────────────────────────
DEVICE="${DEVICE:-}"
if [[ -z "$DEVICE" ]]; then
    DEVICE=$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || { echo "WARNING: torch not importable, defaulting device to 'cpu'" >&2; echo "cpu"; })
fi
echo "Using device: $DEVICE"

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

    log "[PREPARE] Shakespeare (tested dataset)"
    python data/shakespeare/prepare.py

    log "[PREPARE] Wikitext-2 (novel dataset 1)"
    python data/wikitext2/prepare.py

    log "[PREPARE] TinyStories (novel dataset 2)"
    python data/tinystories/prepare.py

fi

# =============================================================================
# TRACK 2 – TRAINING FROM SCRATCH
# =============================================================================
if [[ $DO_TRAIN -eq 1 ]]; then

    # ---------- 2a. Tested dataset: Shakespeare -----------------------------
    log "[TRAIN] CRATE-small on Shakespeare (tested dataset)"
    python train.py config/train_shakespeare_small.py \
        --device="$DEVICE"

    log "[EVAL]  CRATE-small on Shakespeare"
    python train.py config/eval_shakespeare_small.py \
        --out_dir=out-crate-small-shakespeare \
        --device="$DEVICE"

    # ---------- 2b. Novel dataset 1: Wikitext-2 ----------------------------
    log "[TRAIN] CRATE-small on Wikitext-2 (novel dataset 1)"
    python train.py config/train_wikitext2_small.py \
        --device="$DEVICE"

    log "[EVAL]  CRATE-small on Wikitext-2"
    python train.py config/eval_wikitext2_small.py \
        --out_dir=out-crate-small-wikitext2 \
        --device="$DEVICE"

    # ---------- 2c. Novel dataset 2: TinyStories ----------------------------
    log "[TRAIN] CRATE-small on TinyStories (novel dataset 2)"
    python train.py config/train_tinystories_small.py \
        --device="$DEVICE"

    log "[EVAL]  CRATE-small on TinyStories"
    python train.py config/eval_tinystories_small.py \
        --out_dir=out-crate-small-tinystories \
        --device="$DEVICE"

fi

# =============================================================================
# TRACK 3 – FINE-TUNING FROM SHAKESPEARE CHECKPOINT
# =============================================================================
if [[ $DO_FINETUNE -eq 1 ]]; then

    SHAKESPEARE_CKPT="out-crate-small-shakespeare"

    if [[ ! -f "${SHAKESPEARE_CKPT}/ckpt.pt" ]]; then
        echo ""
        echo "WARNING: No Shakespeare checkpoint found at ${SHAKESPEARE_CKPT}/ckpt.pt"
        echo "         Skipping fine-tuning track."
        echo "         Run the training track first (or --train-only), then re-run"
        echo "         with --finetune-only."
        echo ""
    else

        # ---------- 3a. Fine-tune on Wikitext-2 (novel dataset 1) ----------
        log "[FINETUNE] CRATE-small Shakespeare→Wikitext-2"
        python train.py config/finetune_wikitext2_small.py \
            --out_dir="${SHAKESPEARE_CKPT}" \
            --device="$DEVICE"

        log "[EVAL]     CRATE-small fine-tuned on Wikitext-2"
        python train.py config/eval_wikitext2_small.py \
            --out_dir=out-crate-small-ft-wikitext2 \
            --device="$DEVICE"

        # ---------- 3b. Fine-tune on TinyStories (novel dataset 2) ---------
        log "[FINETUNE] CRATE-small Shakespeare→TinyStories"
        python train.py config/finetune_tinystories_small.py \
            --out_dir="${SHAKESPEARE_CKPT}" \
            --device="$DEVICE"

        log "[EVAL]     CRATE-small fine-tuned on TinyStories"
        python train.py config/eval_tinystories_small.py \
            --out_dir=out-crate-small-ft-tinystories \
            --device="$DEVICE"

    fi

fi

log "Small-model benchmark complete."
echo ""
echo "Summary of output directories:"
echo "  out-crate-small-shakespeare/      – CRATE-small trained from scratch on Shakespeare"
echo "  out-crate-small-wikitext2/        – CRATE-small trained from scratch on Wikitext-2"
echo "  out-crate-small-tinystories/      – CRATE-small trained from scratch on TinyStories"
echo "  out-crate-small-ft-wikitext2/     – CRATE-small fine-tuned (Shakespeare→Wikitext-2)"
echo "  out-crate-small-ft-tinystories/   – CRATE-small fine-tuned (Shakespeare→TinyStories)"
echo ""
echo "Perplexity = exp(val_loss).  Collect from the [EVAL] output lines above."
