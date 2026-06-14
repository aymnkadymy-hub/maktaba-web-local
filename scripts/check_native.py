import sys
sys.path.insert(0, r'C:\Users\Ayman\Desktop\book-companion-ai\native_engine\build_win')
sys.path.insert(0, r'C:\Users\Ayman\Desktop\book-companion-ai')

print("=== 1. native_engine C++ module ===")
try:
    import native_engine as ne
    opt = ne.S24DocumentOptimizer()
    eng = ne.AymanComputeEngine(2)
    mc  = ne.MemoryCompressor()
    br  = ne.UniversalModelBridge()
    print("  S24DocumentOptimizer : OK")
    print("  AymanComputeEngine   : OK")
    print("  MemoryCompressor     : OK")
    print("  UniversalModelBridge : OK (stub)")
except Exception as e:
    print(f"  FAIL: {e}")

print()
print("=== 2. ONNX embeddings (Python path) ===")
try:
    from backend.rag.native_embeddings import NativeEmbeddings
    emb = NativeEmbeddings()
    import time
    t = time.perf_counter()
    v = emb.embed_query("اختبار الأداء")
    ms = (time.perf_counter()-t)*1000
    print(f"  embed_query  : OK — dim={len(v)}, {ms:.0f}ms")
    t2 = time.perf_counter()
    vs = emb.embed_documents(["نص 1", "نص 2", "نص 3", "نص 4", "نص 5"])
    ms2 = (time.perf_counter()-t2)*1000
    print(f"  embed_batch  : OK — {len(vs)} vecs, {ms2:.0f}ms")
    print(f"  batch mode   : {'dynamic (fast)' if emb._batch_ok else 'sequential (slow)'}")
except Exception as e:
    print(f"  FAIL: {e}")

print()
print("=== 3. Is server using ONNX? ===")
try:
    from backend.database.vector_db import embeddings
    print(f"  Active embedder: {type(embeddings).__name__}")
except Exception as e:
    print(f"  Could not check: {e}")

print()
print("=== 4. UniversalModelBridge run_inference ===")
try:
    import native_engine as ne
    import numpy as np
    br = ne.UniversalModelBridge()
    br.load_model("test.onnx")
    inp = np.ones(384, dtype=np.float32)
    out = br.run_inference(inp, 384)
    print("  run_inference: OK")
except RuntimeError as e:
    print(f"  run_inference: STUB (expected) — {e}")
except Exception as e:
    print(f"  run_inference: ERROR — {e}")
