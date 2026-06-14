import sys
import time
sys.path.insert(0, r'C:\Users\Ayman\Desktop\book-companion-ai')
from backend.rag.native_embeddings import try_build_native_embeddings
emb = try_build_native_embeddings()
if emb:
    t = time.perf_counter()
    v = emb.embed_query('مرحبا كيف حالك')
    ms = (time.perf_counter()-t)*1000
    print(f'ONNX: WORKING dim={len(v)} time={ms:.0f}ms')
    t2 = time.perf_counter()
    vs = emb.embed_documents(['نص اول', 'نص ثاني', 'نص ثالث'])
    ms2 = (time.perf_counter()-t2)*1000
    print(f'ONNX batch: {len(vs)} vecs time={ms2:.0f}ms')
else:
    print('ONNX: FAILED')
