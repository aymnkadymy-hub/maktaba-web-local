"""
Export paraphrase-multilingual-MiniLM-L12-v2 to ONNX.

Two targets:
  server  → native_engine/models/embedding.onnx  (dynamic batch, mean-pool inside model)
  mobile  → assets/embedding.onnx                (optimum export, for Flutter app)

Usage:
  python scripts/export_embedding_onnx.py           # server model (default)
  python scripts/export_embedding_onnx.py --mobile  # mobile model (requires optimum)
"""

import sys
import shutil
from pathlib import Path

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent

SERVER_DIR  = PROJECT_ROOT / "native_engine" / "models"
SERVER_ONNX = SERVER_DIR / "embedding.onnx"

MOBILE_DIR  = PROJECT_ROOT / "assets"
MOBILE_ONNX = MOBILE_DIR / "embedding.onnx"
QUANT_FILE  = MOBILE_DIR / "embedding_int8.onnx"


# ── Server export ─────────────────────────────────────────────────────────────

def export_server():
    """
    Export for the FastAPI server:
      - Dynamic batch + seq axes on inputs AND output
      - Mean-pool + L2-norm baked into the graph  →  output is sentence_embedding [batch, 384]
      - max_length=512  →  full Arabic paragraph coverage
    """
    print(f"[1/2] Exporting server ONNX → {SERVER_ONNX} ...")
    SERVER_DIR.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModel.from_pretrained(MODEL_ID)
    model.eval()

    # Use a batch-2 sample so torch.onnx traces the batch dimension dynamically
    sample = tokenizer(
        ["مرحباً بالعالم", "Hello world"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    class _Wrapped(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            h  = self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            mk = attention_mask.unsqueeze(-1).float()
            e  = (h * mk).sum(1) / mk.sum(1).clamp(min=1e-9)
            return torch.nn.functional.normalize(e, p=2, dim=1)

    with torch.no_grad():
        torch.onnx.export(
            _Wrapped(model),
            (sample["input_ids"], sample["attention_mask"]),
            str(SERVER_ONNX),
            input_names  = ["input_ids", "attention_mask"],
            output_names = ["sentence_embedding"],
            dynamic_axes = {
                "input_ids":          {0: "batch", 1: "seq"},
                "attention_mask":     {0: "batch", 1: "seq"},
                "sentence_embedding": {0: "batch"},
            },
            opset_version=17,
        )

    sz = SERVER_ONNX.stat().st_size // 1024 // 1024
    print(f"[OK] {sz} MB — {SERVER_ONNX}")
    return True


def verify_server():
    """Verify server ONNX supports batch > 1 and produces unit-norm vectors."""
    print("[2/2] Verifying server ONNX ...")
    import onnxruntime as ort
    from transformers import AutoTokenizer

    sess = ort.InferenceSession(str(SERVER_ONNX), providers=["CPUExecutionProvider"])
    tok  = AutoTokenizer.from_pretrained(MODEL_ID)

    # Check output shape and output name
    out_info = sess.get_outputs()[0]
    print(f"  Output: name={out_info.name!r}, shape={out_info.shape}")
    assert out_info.name == "sentence_embedding", f"Unexpected output name: {out_info.name}"

    # batch=1
    enc1 = tok(["مرحباً"], return_tensors="np", padding=True, truncation=True, max_length=512)
    v1   = sess.run(None, {"input_ids":      enc1["input_ids"].astype("int64"),
                           "attention_mask": enc1["attention_mask"].astype("int64")})[0]
    print(f"  batch=1: shape={v1.shape}, norm={float((v1[0]**2).sum()**0.5):.6f}")
    assert v1.shape == (1, 384), f"Wrong shape: {v1.shape}"

    # batch=4 — this is the critical test for dynamic output batch
    enc4 = tok(["مرحباً", "hello", "TCP/IP", "شبكات"],
               return_tensors="np", padding=True, truncation=True, max_length=512)
    v4   = sess.run(None, {"input_ids":      enc4["input_ids"].astype("int64"),
                           "attention_mask": enc4["attention_mask"].astype("int64")})[0]
    print(f"  batch=4: shape={v4.shape}, norm={float((v4[0]**2).sum()**0.5):.6f}")
    assert v4.shape == (4, 384), f"Wrong shape: {v4.shape}"

    print("[OK] Dynamic batch confirmed — server ONNX ready.")


# ── Mobile export ─────────────────────────────────────────────────────────────

def export_mobile_optimum():
    """Export for Flutter app using optimum (last_hidden_state output)."""
    print(f"[1/3] Mobile export via optimum → {MOBILE_ONNX} ...")
    MOBILE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from optimum.exporters.onnx import main_export
        tmp = MOBILE_DIR / "onnx_tmp"
        main_export(MODEL_ID, output=str(tmp), task="feature-extraction", opset=17)
        shutil.copy(tmp / "model.onnx", MOBILE_ONNX)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[OK] {MOBILE_ONNX.stat().st_size // 1024 // 1024} MB")
        return True
    except Exception as e:
        print(f"[WARN] optimum failed: {e}")
        return False


def quantize_int8():
    print(f"[2/3] INT8 quantization → {QUANT_FILE} ...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(str(MOBILE_ONNX), str(QUANT_FILE), weight_type=QuantType.QInt8)
        print(f"[OK] {QUANT_FILE.stat().st_size // 1024 // 1024} MB")
    except Exception as e:
        print(f"[WARN] Quantization skipped: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export embedding ONNX model")
    parser.add_argument("--mobile", action="store_true",
                        help="Export mobile model to assets/ (requires optimum)")
    args = parser.parse_args()

    if args.mobile:
        ok = export_mobile_optimum()
        if ok:
            quantize_int8()
            print("\nNext: copy assets/embedding_int8.onnx → mobile/assets/embedding.onnx")
    else:
        try:
            import torch  # noqa: F401
        except ImportError:
            print("[ERROR] PyTorch not installed.")
            print("Run: pip install torch --index-url https://download.pytorch.org/whl/cpu")
            sys.exit(1)
        export_server()
        verify_server()
        print("\nServer ONNX updated. Restart the server — batch inference will be active.")
