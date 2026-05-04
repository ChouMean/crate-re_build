param(
    [switch]$PrepareOnly,
    [switch]$TrainOnly,
    [switch]$FinetuneOnly
)

$ErrorActionPreference = "Stop"

function Log-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message"
    Write-Host ""
}

if (-not $env:DEVICE -or $env:DEVICE.Trim() -eq "") {
    try {
        $env:DEVICE = python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"
    }
    catch {
        Write-Warning "torch not importable, defaulting DEVICE to cpu"
        $env:DEVICE = "cpu"
    }
}
Write-Host "Using device: $env:DEVICE"

$doPrepare = $true
$doTrain = $true
$doFinetune = $true

if ($PrepareOnly) {
    $doTrain = $false
    $doFinetune = $false
}
if ($TrainOnly) {
    $doPrepare = $false
    $doFinetune = $false
}
if ($FinetuneOnly) {
    $doPrepare = $false
    $doTrain = $false
}

Log-Step "Checking Python dependencies"
python -c "import importlib.util, sys; pkgs=['torch','tiktoken','datasets','tqdm','requests']; missing=[p for p in pkgs if importlib.util.find_spec(p) is None]; print('ERROR: missing packages: ' + ', '.join(missing)) if missing else print('All dependencies present.'); sys.exit(1 if missing else 0)"

if ($doPrepare) {
    Log-Step "[PREPARE] Shakespeare (tested dataset)"
    python data/shakespeare/prepare.py

    Log-Step "[PREPARE] Wikitext-2 (novel dataset 1)"
    python data/wikitext2/prepare.py

    Log-Step "[PREPARE] TinyStories (novel dataset 2)"
    python data/tinystories/prepare.py
}

if ($doTrain) {
    Log-Step "[TRAIN] CRATE-small on Shakespeare (tested dataset)"
    python train.py config/train_shakespeare_small.py --device="$env:DEVICE"

    Log-Step "[EVAL] CRATE-small on Shakespeare"
    python train.py config/eval_shakespeare_small.py --out_dir=out-crate-small-shakespeare --device="$env:DEVICE"

    Log-Step "[TRAIN] CRATE-small on Wikitext-2 (novel dataset 1)"
    python train.py config/train_wikitext2_small.py --device="$env:DEVICE"

    Log-Step "[EVAL] CRATE-small on Wikitext-2"
    python train.py config/eval_wikitext2_small.py --out_dir=out-crate-small-wikitext2 --device="$env:DEVICE"

    Log-Step "[TRAIN] CRATE-small on TinyStories (novel dataset 2)"
    python train.py config/train_tinystories_small.py --device="$env:DEVICE"

    Log-Step "[EVAL] CRATE-small on TinyStories"
    python train.py config/eval_tinystories_small.py --out_dir=out-crate-small-tinystories --device="$env:DEVICE"
}

if ($doFinetune) {
    $shakespeareCkpt = "out-crate-small-shakespeare"
    $ckptPath = Join-Path $shakespeareCkpt "ckpt.pt"
    $wikitextFtDir = "out-crate-small-ft-wikitext2"
    $tinystoriesFtDir = "out-crate-small-ft-tinystories"

    if (-not (Test-Path $ckptPath)) {
        Write-Host ""
        Write-Warning "No Shakespeare checkpoint found at $ckptPath"
        Write-Host "Skipping fine-tuning track."
        Write-Host "Run the training track first, then re-run with -FinetuneOnly."
        Write-Host ""
    }
    else {
        Log-Step "[SETUP] Copy Shakespeare checkpoint for Wikitext-2 fine-tuning"
        New-Item -ItemType Directory -Force $wikitextFtDir | Out-Null
        Copy-Item $ckptPath (Join-Path $wikitextFtDir "ckpt.pt") -Force

        Log-Step "[FINETUNE] CRATE-small Shakespeare to Wikitext-2"
        python train.py config/finetune_wikitext2_small.py --out_dir="$wikitextFtDir" --device="$env:DEVICE"

        Log-Step "[EVAL] CRATE-small fine-tuned on Wikitext-2"
        python eval_checkpoint.py --out_dir="$wikitextFtDir" --dataset=wikitext2 --device="$env:DEVICE" --eval_iters=50

        Log-Step "[SETUP] Copy Shakespeare checkpoint for TinyStories fine-tuning"
        New-Item -ItemType Directory -Force $tinystoriesFtDir | Out-Null
        Copy-Item $ckptPath (Join-Path $tinystoriesFtDir "ckpt.pt") -Force

        Log-Step "[FINETUNE] CRATE-small Shakespeare to TinyStories"
        python train.py config/finetune_tinystories_small.py --out_dir="$tinystoriesFtDir" --device="$env:DEVICE"

        Log-Step "[EVAL] CRATE-small fine-tuned on TinyStories"
        python eval_checkpoint.py --out_dir="$tinystoriesFtDir" --dataset=tinystories --device="$env:DEVICE" --eval_iters=50
    }
}

Log-Step "Small-model benchmark complete."
Write-Host ""
Write-Host "Summary of output directories:"
Write-Host "  out-crate-small-shakespeare/"
Write-Host "  out-crate-small-wikitext2/"
Write-Host "  out-crate-small-tinystories/"
Write-Host "  out-crate-small-ft-wikitext2/"
Write-Host "  out-crate-small-ft-tinystories/"
Write-Host ""
Write-Host "Perplexity = exp(val_loss). Collect from the [EVAL] output lines above."
