# setup_win.ps1 — First-time setup for المكتبة الناطقة (Local Windows Server)
# Run once from the project root: .\scripts\setup_win.ps1
# Requires: Python 3.11+, internet connection

#Requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent $PSScriptRoot
Push-Location $ROOT

function Write-Step([string]$msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-OK([string]$msg)   { Write-Host "    [OK] $msg" -ForegroundColor Green }
function Write-Warn([string]$msg) { Write-Host "    [WARN] $msg" -ForegroundColor Yellow }
function Write-Fail([string]$msg) { Write-Host "    [ERR] $msg" -ForegroundColor Red; Pop-Location; exit 1 }

Write-Host ""
Write-Host "╔══════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   المكتبة الناطقة — Local Server Setup      ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════╝" -ForegroundColor Cyan

# ── Step 1: Python check ───────────────────────────────────────────────────────
Write-Step "Step 1/6: Checking Python version"
try {
    $ver = python --version 2>&1
    if ($ver -notmatch "3\.(11|12|13)") {
        Write-Warn "Python $ver found. Python 3.11+ recommended."
    } else {
        Write-OK $ver
    }
} catch {
    Write-Fail "Python not found. Install Python 3.11+ from python.org"
}

# ── Step 2: Virtual environment ────────────────────────────────────────────────
Write-Step "Step 2/6: Setting up .venv_win"
if (-not (Test-Path ".venv_win")) {
    Write-Host "    Creating .venv_win ..."
    python -m venv .venv_win
    Write-OK "Created .venv_win"
} else {
    Write-OK ".venv_win already exists"
}

$PIP = ".venv_win\Scripts\pip.exe"
$PY  = ".venv_win\Scripts\python.exe"

Write-Host "    Upgrading pip..."
& $PY -m pip install --upgrade pip --quiet
& $PIP install --upgrade wheel --quiet

# ── Step 3: Dependencies ───────────────────────────────────────────────────────
Write-Step "Step 3/6: Installing Python dependencies"
& $PIP install -r requirements.txt --quiet
if ($LASTEXITCODE -ne 0) { Write-Fail "pip install failed" }
Write-OK "All Python dependencies installed"

# ── Step 4: Qdrant binary ──────────────────────────────────────────────────────
Write-Step "Step 4/6: Qdrant vector database binary"
$QDRANT_EXE = "tools\qdrant.exe"
if (-not (Test-Path $QDRANT_EXE)) {
    Write-Host "    Downloading Qdrant v1.9.4 for Windows..."
    New-Item -ItemType Directory -Force -Path "tools" | Out-Null
    $QDRANT_URL = "https://github.com/qdrant/qdrant/releases/download/v1.9.4/qdrant-x86_64-pc-windows-msvc.zip"
    $ZIP_PATH   = "tools\qdrant.zip"
    try {
        Invoke-WebRequest -Uri $QDRANT_URL -OutFile $ZIP_PATH -UseBasicParsing
        Expand-Archive -Path $ZIP_PATH -DestinationPath "tools" -Force
        Remove-Item $ZIP_PATH -Force
        # Find extracted exe
        $exe = Get-ChildItem "tools" -Recurse -Filter "qdrant.exe" | Select-Object -First 1
        if ($exe -and $exe.FullName -ne (Resolve-Path $QDRANT_EXE -ErrorAction SilentlyContinue)) {
            Move-Item $exe.FullName $QDRANT_EXE -Force
        }
        Write-OK "Qdrant downloaded: $QDRANT_EXE"
    } catch {
        Write-Warn "Auto-download failed: $_"
        Write-Warn "Download manually from: https://github.com/qdrant/qdrant/releases"
        Write-Warn "Place qdrant.exe in: $(Resolve-Path tools)"
    }
} else {
    Write-OK "Qdrant already present: $QDRANT_EXE"
}

# Create Qdrant config if missing
New-Item -ItemType Directory -Force -Path "config" | Out-Null
if (-not (Test-Path "config\qdrant_config.yaml")) {
    @"
storage:
  storage_path: ./qdrant_data
service:
  host: 0.0.0.0
  http_port: 6333
  grpc_port: 6334
log_level: WARN
"@ | Set-Content "config\qdrant_config.yaml" -Encoding UTF8
    Write-OK "config/qdrant_config.yaml created"
}

# ── Step 5: Embedding model ────────────────────────────────────────────────────
Write-Step "Step 5/6: Embedding ONNX model"
$EMBEDDING = "native_engine\models\embedding.onnx"
if (-not (Test-Path $EMBEDDING)) {
    Write-Host "    Exporting embedding ONNX (requires PyTorch)..."
    & $PY scripts\export_embedding_onnx.py
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Export failed — install PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu"
    } else {
        Write-OK $EMBEDDING
    }
} else {
    Write-OK "$EMBEDDING already present"
}

# ── Step 6: .env file ──────────────────────────────────────────────────────────
Write-Step "Step 6/6: Environment configuration"
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-OK "Created .env from .env.example"
        Write-Warn "Edit .env to set your API keys (optional)"
    } else {
        @"
LLM_PROVIDER=ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
EMBEDDING_MODEL=native_engine/models/embedding.onnx
BOOKS_DIR=books
QDRANT_PATH=qdrant_data
BM25_CACHE_DIR=bm25_cache
ENABLE_RERANKER=true
NUM_CTX=4096
NUM_PREDICT=300
TEMPERATURE=0.2
RAG_K=5
"@ | Set-Content ".env" -Encoding UTF8
        Write-OK ".env created with defaults"
    }
} else {
    Write-OK ".env already exists"
}

# ── Data directories ───────────────────────────────────────────────────────────
New-Item -ItemType Directory -Force -Path "books","qdrant_data","bm25_cache","logs" | Out-Null

Write-Host ""
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Green
Write-Host " Setup complete! Next steps:" -ForegroundColor Green
Write-Host ""
Write-Host "  1. Install Ollama: https://ollama.com/download" -ForegroundColor White
Write-Host "  2. Pull model:  ollama pull qwen2.5:3b" -ForegroundColor White
Write-Host "  3. Start server: double-click تشغيل_الموقع.bat" -ForegroundColor White
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Green
Pop-Location
