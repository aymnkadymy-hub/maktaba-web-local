# download_models.ps1 — Download embedding ONNX + GGUF LLM from HuggingFace
# Usage:
#   .\scripts\download_models.ps1                  # embedding + LLM
#   .\scripts\download_models.ps1 -EmbeddingOnly   # embedding model only
#   .\scripts\download_models.ps1 -SkipLLM         # same as -EmbeddingOnly
#
# Requirements: Python 3.11+ in .venv_win (or system Python with pip)

param(
    [switch]$EmbeddingOnly,
    [switch]$SkipLLM,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent $PSScriptRoot
Push-Location $ROOT

function Write-Step([string]$msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-OK([string]$msg)   { Write-Host "    [OK] $msg" -ForegroundColor Green }
function Write-Warn([string]$msg) { Write-Host "    [WARN] $msg" -ForegroundColor Yellow }

# ── Resolve Python ─────────────────────────────────────────────────────────────
$PY = if (Test-Path ".venv_win\Scripts\python.exe") {
    ".venv_win\Scripts\python.exe"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    "python"
} else {
    throw "Python not found. Run setup_win.ps1 first."
}
Write-Step "Using Python: $PY"

# ── Paths ──────────────────────────────────────────────────────────────────────
$NATIVE_MODELS = "native_engine\models"
$MOBILE_ASSETS = "mobile\assets"
$MODELS_DIR    = "mobile\assets\models"

New-Item -ItemType Directory -Force -Path $NATIVE_MODELS | Out-Null
New-Item -ItemType Directory -Force -Path $MOBILE_ASSETS | Out-Null
New-Item -ItemType Directory -Force -Path $MODELS_DIR    | Out-Null

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Embedding ONNX (paraphrase-multilingual-MiniLM-L12-v2)
# ══════════════════════════════════════════════════════════════════════════════
Write-Step "Part 1/2: Embedding ONNX model"

$serverOnnx = "$NATIVE_MODELS\embedding.onnx"
$mobileOnnx = "$MOBILE_ASSETS\embedding.onnx"

if ((Test-Path $serverOnnx) -and (Test-Path $mobileOnnx) -and -not $Force) {
    Write-OK "Embedding models already present (use -Force to re-download)"
} else {
    Write-Host "    Exporting via scripts\export_embedding_onnx.py ..."

    # Server ONNX
    & $PY scripts\export_embedding_onnx.py
    if ($LASTEXITCODE -ne 0) { throw "Server ONNX export failed" }
    Write-OK "native_engine/models/embedding.onnx"

    # Mobile ONNX (optimum export)
    & $PY scripts\export_embedding_onnx.py --mobile
    if ($LASTEXITCODE -ne 0) {
        # Fallback: copy server ONNX to mobile assets
        Write-Warn "Optimum export failed — copying server model to mobile assets"
        Copy-Item $serverOnnx $mobileOnnx -Force
        if (Test-Path "$NATIVE_MODELS\embedding.onnx.data") {
            Copy-Item "$NATIVE_MODELS\embedding.onnx.data" "$MOBILE_ASSETS\embedding.onnx.data" -Force
        }
    } else {
        # copy result from assets/ (export script writes there)
        $exportedMobile = "assets\embedding.onnx"
        if (Test-Path $exportedMobile) {
            Copy-Item $exportedMobile $mobileOnnx -Force
            Write-OK "mobile/assets/embedding.onnx (optimum)"
        } elseif (-not (Test-Path $mobileOnnx)) {
            Copy-Item $serverOnnx $mobileOnnx -Force
            Write-OK "mobile/assets/embedding.onnx (fallback from server model)"
        }
    }
}

# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — GGUF LLM  (qwen2.5-3b-instruct-q4_k_m.gguf ~2.1 GB)
# ══════════════════════════════════════════════════════════════════════════════
if ($EmbeddingOnly -or $SkipLLM) {
    Write-Warn "Skipping GGUF download (-EmbeddingOnly / -SkipLLM)"
    Pop-Location
    exit 0
}

Write-Step "Part 2/2: GGUF LLM (qwen2.5-3b-instruct-q4_k_m.gguf ~2.1 GB)"

$GGUF_FILE  = "qwen2.5-3b-instruct-q4_k_m.gguf"
$GGUF_DEST  = "$MODELS_DIR\$GGUF_FILE"
$GGUF_URL   = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/$GGUF_FILE"

if ((Test-Path $GGUF_DEST) -and -not $Force) {
    $size = (Get-Item $GGUF_DEST).Length
    Write-OK "$GGUF_FILE already present ($([math]::Round($size/1MB, 0)) MB)"
} else {
    Write-Host "    Downloading from HuggingFace (this may take several minutes)..."
    Write-Host "    URL: $GGUF_URL"
    Write-Host "    Dest: $GGUF_DEST"

    # Use BitsTransfer if available, else WebClient with progress
    try {
        Import-Module BitsTransfer -ErrorAction Stop
        Start-BitsTransfer -Source $GGUF_URL -Destination $GGUF_DEST `
            -DisplayName "Downloading $GGUF_FILE" -Description "LLM for offline AI"
        Write-OK "Downloaded via BITS: $GGUF_DEST"
    } catch {
        Write-Warn "BITS unavailable — using WebClient (no progress bar)"
        $wc = New-Object System.Net.WebClient
        $wc.DownloadFile($GGUF_URL, (Resolve-Path ".").Path + "\$GGUF_DEST")
        Write-OK "Downloaded: $GGUF_DEST"
    }
}

# ── Desktop Windows App models folder ─────────────────────────────────────────
$WIN_APP_MODELS = "$env:USERPROFILE\Desktop\maktaba-windows-app\models"
if (Test-Path $WIN_APP_MODELS) {
    $WIN_GGUF = "$WIN_APP_MODELS\$GGUF_FILE"
    if (-not (Test-Path $WIN_GGUF)) {
        Write-Host "    Copying GGUF to maktaba-windows-app\models\ ..."
        Copy-Item $GGUF_DEST $WIN_GGUF
        Write-OK "Copied to Windows App workspace"
    }
}

Write-Host ""
Write-Host "════════════════════════════════════════════" -ForegroundColor Green
Write-Host " Download complete!" -ForegroundColor Green
Write-Host "  Embedding: $mobileOnnx" -ForegroundColor Green
Write-Host "  LLM GGUF:  $GGUF_DEST" -ForegroundColor Green
Write-Host "════════════════════════════════════════════" -ForegroundColor Green

Pop-Location
