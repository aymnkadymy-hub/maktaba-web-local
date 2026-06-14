"""
تصدير BAAI/bge-m3 إلى صيغة ONNX.

بالمقارنة مع paraphrase-multilingual-MiniLM-L12-v2 الحالي:
  ┌──────────────────────────────────┬─────────┬──────────┐
  │ Model                            │ Dim     │ MTEB-AR  │
  ├──────────────────────────────────┼─────────┼──────────┤
  │ paraphrase-multilingual-MiniLM   │  384    │  ~44     │
  │ BAAI/bge-m3  (هذا الملف)        │ 1024    │  ~57     │
  └──────────────────────────────────┴─────────┴──────────┘

تحذير:
  - حجم النموذج: ~570 MB
  - يحتاج إعادة استيعاب (re-ingest) جميع الكتب بعد التبديل.
  - حذف مجلد chroma_data/ قبل التشغيل لتفادي تضارب الأبعاد.

تشغيل:
  python scripts/export_bge_m3_onnx.py

بعد النجاح:
  1. احذف:  chroma_data/
  2. غيّر ONNX_MODEL في native_embeddings.py إلى المسار الجديد.
  3. غيّر MODEL_NAME إلى "BAAI/bge-m3"
  4. أعد تشغيل الخادم — سيُعيد استيعاب الكتب تلقائياً.
"""
import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "native_engine", "models")
OUTPUT_PATH  = os.path.join(MODELS_DIR, "bge_m3.onnx")
MODEL_NAME   = "BAAI/bge-m3"

os.makedirs(MODELS_DIR, exist_ok=True)

print(f"تحميل النموذج: {MODEL_NAME}  (~570 MB — قد يستغرق وقتاً)")
from sentence_transformers import SentenceTransformer
st_model   = SentenceTransformer(MODEL_NAME)
transformer = st_model[0]
auto_model  = transformer.auto_model
tokenizer   = transformer.tokenizer
auto_model.eval()

print("إعداد المدخلات الوهمية للتصدير...")
dummy = tokenizer(
    "This is a test sentence for ONNX export of bge-m3.",
    return_tensors="pt",
    padding="max_length",
    max_length=512,
    truncation=True,
)
input_ids      = dummy["input_ids"]
attention_mask = dummy["attention_mask"]

print(f"تصدير ONNX إلى: {OUTPUT_PATH}")
with torch.no_grad():
    torch.onnx.export(
        auto_model,
        (input_ids, attention_mask),
        OUTPUT_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids":         {0: "batch", 1: "seq_len"},
            "attention_mask":    {0: "batch", 1: "seq_len"},
            "last_hidden_state": {0: "batch", 1: "seq_len"},
            "pooler_output":     {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

print("التحقق من الملف...")
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession(OUTPUT_PATH, providers=["CPUExecutionProvider"])
inp  = tokenizer("اختبار النموذج", return_tensors="np",
                 padding=True, truncation=True, max_length=512)
out  = sess.run(None, {
    "input_ids":      inp["input_ids"].astype(np.int64),
    "attention_mask": inp["attention_mask"].astype(np.int64),
})
hidden = out[0]   # (1, seq_len, 1024)
print(f"شكل الناتج: {hidden.shape}  →  embedding dim = {hidden.shape[-1]}")
assert hidden.shape[-1] == 1024, "الأبعاد خاطئة!"
print(f"✅ تم التصدير بنجاح: {OUTPUT_PATH}")
print(f"   حجم الملف: {os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.1f} MB")

print("""
─────────────────────────────────────────────────────
الخطوات التالية لتفعيل bge-m3:

1. احذف مجلد chroma_data/ (ضروري — تغيّرت الأبعاد 384→1024)

2. في backend/rag/native_embeddings.py غيّر:
     ONNX_MODEL = "...native_engine/models/bge_m3.onnx"
     MODEL_NAME  = "BAAI/bge-m3"
     MAX_LENGTH  = 512

3. أعد تشغيل الخادم — سيستوعب الكتب تلقائياً.
─────────────────────────────────────────────────────
""")
