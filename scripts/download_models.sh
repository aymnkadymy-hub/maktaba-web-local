#!/usr/bin/env bash
# download_models.sh — Download embedding ONNX + GGUF LLM from HuggingFace
# Usage:
#   bash scripts/download_models.sh              # embedding + LLM
#   bash scripts/download_models.sh --embedding  # embedding only
#   bash scripts/download_models.sh --force      # force re-download
#
# Requirements: Python 3.11+ with venv at .venv (or system python)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

EMBEDDING_ONLY=0
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --embedding|-e) EMBEDDING_ONLY=1 ;;
        --force|-f)     FORCE=1 ;;
    esac
done

step() { echo; echo "==> $1"; }
ok()   { echo "    [OK] $1"; }
warn() { echo "    [WARN] $1"; }

# ── Resolve Python ─────────────────────────────────────────────────────────────
if [ -f ".venv/bin/python" ]; then
    PY=".venv/bin/python"
elif [ -f "venv_new/bin/python" ]; then
    PY="venv_new/bin/python"
elif command -v python3 &>/dev/null; then
    PY="python3"
elif command -v python &>/dev/null; then
    PY="python"
else
    echo "[ERROR] Python not found. Install Python 3.11+ first."
    exit 1
fi
step "Using Python: $PY"

# ── Paths ──────────────────────────────────────────────────────────────────────
NATIVE_MODELS="native_engine/models"
MOBILE_ASSETS="mobile/assets"
MODELS_DIR="mobile/assets/models"

mkdir -p "$NATIVE_MODELS" "$MOBILE_ASSETS" "$MODELS_DIR"

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Embedding ONNX
# ══════════════════════════════════════════════════════════════════════════════
step "Part 1/2: Embedding ONNX model"

SERVER_ONNX="$NATIVE_MODELS/embedding.onnx"
MOBILE_ONNX="$MOBILE_ASSETS/embedding.onnx"

if [ -f "$SERVER_ONNX" ] && [ -f "$MOBILE_ONNX" ] && [ "$FORCE" -eq 0 ]; then
    ok "Embedding models already present (use --force to re-export)"
else
    echo "    Exporting via scripts/export_embedding_onnx.py ..."

    # Server ONNX
    "$PY" scripts/export_embedding_onnx.py || { echo "[ERROR] Server ONNX export failed"; exit 1; }
    ok "$SERVER_ONNX"

    # Mobile ONNX (optimum)
    if "$PY" scripts/export_embedding_onnx.py --mobile 2>/dev/null; then
        # export script writes to assets/embedding.onnx (project root assets/)
        if [ -f "assets/embedding.onnx" ]; then
            cp "assets/embedding.onnx" "$MOBILE_ONNX"
            ok "$MOBILE_ONNX (optimum)"
        fi
    fi

    # Fallback: use server model for mobile
    if [ ! -f "$MOBILE_ONNX" ]; then
        warn "Using server ONNX as fallback for mobile assets"
        cp "$SERVER_ONNX" "$MOBILE_ONNX"
        [ -f "${SERVER_ONNX}.data" ] && cp "${SERVER_ONNX}.data" "${MOBILE_ONNX}.data"
        ok "$MOBILE_ONNX (fallback)"
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — GGUF LLM
# ══════════════════════════════════════════════════════════════════════════════
if [ "$EMBEDDING_ONLY" -eq 1 ]; then
    warn "Skipping GGUF download (--embedding flag)"
    exit 0
fi

step "Part 2/2: GGUF LLM (qwen2.5-3b-instruct-q4_k_m.gguf ~2.1 GB)"

GGUF_FILE="qwen2.5-3b-instruct-q4_k_m.gguf"
GGUF_DEST="$MODELS_DIR/$GGUF_FILE"
GGUF_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/$GGUF_FILE"

if [ -f "$GGUF_DEST" ] && [ "$FORCE" -eq 0 ]; then
    SIZE=$(du -sh "$GGUF_DEST" | cut -f1)
    ok "$GGUF_FILE already present ($SIZE)"
else
    echo "    Downloading from HuggingFace..."
    echo "    URL: $GGUF_URL"
    echo "    Dest: $GGUF_DEST"

    if command -v wget &>/dev/null; then
        wget -c --show-progress -O "$GGUF_DEST" "$GGUF_URL"
    elif command -v curl &>/dev/null; then
        curl -L --continue-at - --progress-bar -o "$GGUF_DEST" "$GGUF_URL"
    elif command -v huggingface-cli &>/dev/null; then
        huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF "$GGUF_FILE" \
            --local-dir "$MODELS_DIR"
    else
        echo "[ERROR] No download tool found (install wget, curl, or huggingface-cli)"
        exit 1
    fi
    ok "Downloaded: $GGUF_DEST"
fi

echo ""
echo "════════════════════════════════════════════"
echo " Download complete!"
echo "  Embedding: $MOBILE_ONNX"
echo "  LLM GGUF:  $GGUF_DEST"
echo "════════════════════════════════════════════"
