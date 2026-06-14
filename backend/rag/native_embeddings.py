"""
NativeEmbeddings: ONNX Runtime embeddings compatible with LangChain.

Pipeline:
  text → HuggingFace tokenizer (Python) → ONNX Runtime inference → sentence vector

Execution provider priority (auto-selected at init):
  1. CUDAExecutionProvider   — RTX 3050 6GB, requires onnxruntime-gpu
  2. CPUExecutionProvider    — always available as fallback

Supports two ONNX model formats transparently:
  Legacy (optimum export):  output = last_hidden_state [batch, seq, 384]  → Python mean-pool + L2-norm
  Server (torch export):    output = sentence_embedding [batch, 384]       → already pooled + normalised

Batch mode:
  If the ONNX output batch dimension is dynamic → true batch (BATCH_SIZE texts per forward pass).
  If the ONNX output batch dimension is static (=1) → sequential (one text per pass).
  Detected at __init__ time — the session's memory arena is never touched with the wrong shape.
"""
import os
import logging
import subprocess
import threading
import numpy as np
from typing import List
from langchain_core.embeddings import Embeddings

logger = logging.getLogger("native_embeddings")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ONNX_MODEL   = os.path.join(PROJECT_ROOT, "native_engine", "models", "embedding.onnx")
MODEL_NAME   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_LENGTH   = 512

# Minimum free VRAM (MB) to attempt CUDA loading
_CUDA_MIN_VRAM_MB = 800


def _free_vram_mb() -> int:
    """Query free VRAM via nvidia-smi. Returns 0 if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            timeout=3, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return int(out.split("\n")[0].strip())
    except Exception:
        return 0


def _gpu_name() -> str:
    """Query GPU name via nvidia-smi. Returns 'GPU' if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            timeout=3, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out.split("\n")[0].strip()
    except Exception:
        return "GPU"


# ── Native engine: C++ thread pool + FP16 compressor ─────────────────────────

_native_pool:       object = None   # AymanComputeEngine singleton
_native_compressor: object = None   # MemoryCompressor singleton


def _init_native_components() -> None:
    """Load AymanComputeEngine + MemoryCompressor from native_engine .pyd if available."""
    global _native_pool, _native_compressor
    try:
        import sys as _sys
        for _p in [
            os.path.join(PROJECT_ROOT, "native_engine", "build_win"),
            os.path.join(PROJECT_ROOT, "native_engine", "build"),
        ]:
            if os.path.isdir(_p):
                if _p not in _sys.path:
                    _sys.path.insert(0, _p)
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(_p)
        import native_engine as _ne
        cpu = max(2, (os.cpu_count() or 4) // 2)
        _native_pool       = _ne.AymanComputeEngine(cpu)
        _native_compressor = _ne.MemoryCompressor()
        logger.info(
            f"native_engine: C++ thread pool ({cpu} threads) + FP16 compressor active"
        )
    except Exception as _e:
        logger.debug(f"native_engine components unavailable: {_e}")


_init_native_components()


def submit_background_task(fn) -> None:
    """
    Schedule fn() on the C++ thread pool (or a daemon thread as fallback).
    Drop-in replacement for threading.Thread(target=fn, daemon=True).start().
    """
    if _native_pool is not None:
        _native_pool.send_to_pipe(fn)
    else:
        import threading
        threading.Thread(target=fn, daemon=True).start()


# ── FP16 embedding cache (MemoryCompressor or numpy fallback) ─────────────────

_EMBED_CACHE_MAXSIZE = 512    # max cached query vectors


class _CompressedEmbedCache:
    """
    FIFO cache for query embeddings stored as FP16.

    Uses MemoryCompressor (C++ AVX2 path) when available, otherwise falls back to
    numpy float16 view — both produce identical IEEE 754 float16 bit patterns.

    384-dim float32 = 1536 B  →  384 uint16 = 768 B  (2× RAM savings).
    FP16 precision loss < 0.1 % for L2-normalised vectors — negligible for cosine search.
    Requirement: embedding dimension must be a multiple of 64 (384 = 6 × 64 ✓).
    """

    def __init__(self, maxsize: int = _EMBED_CACHE_MAXSIZE):
        self._maxsize = maxsize
        self._cache:  dict[str, np.ndarray] = {}   # text → uint16 array
        self._order:  list[str]             = []   # FIFO eviction queue
        self._lock    = threading.Lock()

    @staticmethod
    def _compress(vec: np.ndarray) -> np.ndarray:
        if _native_compressor is not None:
            return _native_compressor.compress(vec)
        return vec.astype(np.float16).view(np.uint16)

    @staticmethod
    def _decompress(data: np.ndarray) -> list:
        if _native_compressor is not None:
            return _native_compressor.decompress(data).tolist()
        return data.view(np.float16).astype(np.float32).tolist()

    def get(self, text: str) -> "list | None":
        with self._lock:
            data = self._cache.get(text)
        return self._decompress(data) if data is not None else None

    def put(self, text: str, embedding: list) -> None:
        compressed = self._compress(np.array(embedding, dtype=np.float32))
        with self._lock:
            if text in self._cache:
                return
            if len(self._cache) >= self._maxsize:
                evict = self._order.pop(0)
                self._cache.pop(evict, None)
            self._cache[text] = compressed
            self._order.append(text)


_embed_cache = _CompressedEmbedCache()


def _probe_cuda_subprocess(onnx_path: str) -> bool:
    """
    Test CUDA provider in a child process to avoid crashing the main server.
    Ollama's cuDNN is incompatible with ORT (wrong symbols) and causes a
    C-level crash that Python try/except cannot catch. Running the test in
    a subprocess isolates the crash.
    Returns True only if a real CUDA session was created successfully.
    """
    import sys
    code = f"""
import os, sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import onnxruntime as ort
# Preload CUDA/cuDNN DLLs from pip packages (nvidia-cublas-cu12, nvidia-cufft-cu12, etc.)
if hasattr(ort, 'preload_dlls'):
    try:
        ort.preload_dlls(cuda=True, cudnn=True)
    except Exception:
        pass
if "CUDAExecutionProvider" not in ort.get_available_providers():
    sys.exit(2)
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
try:
    sess = ort.InferenceSession(
        {onnx_path!r},
        sess_options=opts,
        providers=[("CUDAExecutionProvider", {{"device_id": 0}}), "CPUExecutionProvider"],
    )
    active = sess.get_providers()[0]
    sys.exit(0 if active == "CUDAExecutionProvider" else 1)
except Exception as e:
    sys.exit(1)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            timeout=30, capture_output=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def _probe_dml_subprocess(onnx_path: str) -> bool:
    """Test DirectML provider with real inference in a child process to avoid crashes."""
    import sys
    code = f"""
import os, sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np
import onnxruntime as ort
if "DmlExecutionProvider" not in ort.get_available_providers():
    sys.exit(2)
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
try:
    sess = ort.InferenceSession(
        {onnx_path!r},
        sess_options=opts,
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    if sess.get_providers()[0] != "DmlExecutionProvider":
        sys.exit(1)
    # Test real inference with batch=1 to catch Reshape errors
    dummy = np.ones((1, 16), dtype=np.int64)
    sess.run(None, {{"input_ids": dummy, "attention_mask": dummy}})
    sys.exit(0)
except Exception:
    sys.exit(1)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            timeout=30, capture_output=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def _select_providers(onnx_path: str) -> tuple[list, str]:
    """
    Pick the best available execution provider.
    Priority: CUDA → DirectML → CPU.
    Uses subprocess probes to safely test GPU providers.
    Returns (providers_list, backend_name).
    """
    import onnxruntime as ort

    available = ort.get_available_providers()

    if "CUDAExecutionProvider" in available:
        free_mb = _free_vram_mb()
        if free_mb >= _CUDA_MIN_VRAM_MB:
            logger.info(f"Probing CUDA provider ({free_mb} MB VRAM free) …")
            if _probe_cuda_subprocess(onnx_path):
                cuda_opts = {
                    "device_id": 0,
                    "gpu_mem_limit": int(free_mb * 0.70 * 1024 * 1024),
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "DEFAULT",
                    "do_copy_in_default_stream": True,
                }
                gpu = _gpu_name()
                logger.info(f"CUDA probe passed — using GPU {gpu} (limit {int(free_mb*0.70)} MB)")
                return (
                    [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"],
                    f"CUDA [{gpu}, {free_mb}MB free]",
                )
            else:
                logger.info("CUDA probe failed — trying DirectML next")
        else:
            logger.info(f"CUDA available but only {free_mb} MB VRAM free — trying DirectML")

    if "DmlExecutionProvider" in available:
        logger.info("Probing DirectML provider (DirectX 12 GPU acceleration) …")
        if _probe_dml_subprocess(onnx_path):
            gpu = _gpu_name()
            logger.info(f"DirectML probe passed — using GPU via DirectX 12 ({gpu})")
            return (
                ["DmlExecutionProvider", "CPUExecutionProvider"],
                f"DirectML [{gpu}]",
            )
        else:
            logger.info("DirectML probe failed — falling back to CPU")

    return (["CPUExecutionProvider"], "CPU")


class NativeEmbeddings(Embeddings):
    """
    Drop-in replacement for HuggingFaceEmbeddings.
    Auto-selects CUDA → CPU execution provider at init time.
    CUDA gives 3-8× speedup over CPU for batch embedding inference.
    """

    # Overridden per-instance in __init__ based on detected backend:
    #   CPU      → 256  (no VRAM limit, 16× more throughput per ONNX call)
    #   DirectML → 32   (DX12 GPU, conservative to avoid driver OOM)
    #   CUDA     → 16   (RTX 3050 6 GB — was 32 → OOM on 1000-page books)
    BATCH_SIZE = 16

    def __init__(
        self,
        onnx_path: str = ONNX_MODEL,
        tokenizer_name: str = MODEL_NAME,
        max_length: int = MAX_LENGTH,
    ):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(
                f"ONNX model not found: {onnx_path}\n"
                f"Run:  python scripts/export_embedding_onnx.py"
            )

        # Preload CUDA/cuDNN from pip packages before provider selection
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls(cuda=True, cudnn=True)
            except Exception:
                pass

        self._onnx_path = onnx_path
        providers, self.backend = _select_providers(onnx_path)

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # CPU thread tuning (still used even with CUDA for CPU-fallback ops)
        cpu_threads = max(1, (os.cpu_count() or 2) // 2)
        sess_opts.intra_op_num_threads = cpu_threads
        sess_opts.inter_op_num_threads = 1

        try:
            self._session = ort.InferenceSession(
                onnx_path, sess_options=sess_opts, providers=providers
            )
        except Exception as e:
            logger.warning(f"Failed to load with {self.backend}: {e} — falling back to CPU")
            self._session = ort.InferenceSession(
                onnx_path, sess_options=sess_opts,
                providers=["CPUExecutionProvider"]
            )
            self.backend = "CPU (fallback)"

        # Report the provider that was actually used
        active = self._session.get_providers()[0]
        logger.info(f"NativeEmbeddings session active provider: {active}")

        # Set batch size based on backend — CPU has no VRAM limit so we can use large batches
        if "CUDA" in self.backend:
            self.BATCH_SIZE = 16    # 6 GB VRAM — keep conservative
        elif "DirectML" in self.backend:
            self.BATCH_SIZE = 32    # DX12 GPU
        else:
            self.BATCH_SIZE = 256   # CPU: 16× throughput improvement vs 16

        logger.info(f"NativeEmbeddings batch_size={self.BATCH_SIZE} (backend={self.backend})")

        self._tokenizer  = AutoTokenizer.from_pretrained(tokenizer_name)
        self._max_length = max_length

        # ── Detect output format at init time ──────────────────────────────────
        out       = self._session.get_outputs()[0]
        out_shape = out.shape

        self._pre_pooled = out.name == "sentence_embedding"
        self._out_name   = out.name
        self._batch_ok   = not isinstance(out_shape[0], int)

        mode = "batch=32" if self._batch_ok else "sequential (static batch=1)"
        logger.info(
            f"NativeEmbeddings ready — backend={self.backend}, "
            f"output={self._out_name!r}, pooled={self._pre_pooled}, {mode}"
        )

        # Chat-query priority flag: background batch jobs check this and yield
        # so that a live user query never waits for RAPTOR/ingest embedding batches.
        self._chat_priority = threading.Event()

    # ── LangChain interface ────────────────────────────────────────────────────

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import time as _time
        results: List[List[float]] = []
        batch_size = self.BATCH_SIZE
        i = 0
        while i < len(texts):
            # Yield to live chat queries — wait up to 2s for priority to clear
            if self._chat_priority.is_set():
                _time.sleep(0.002)
                continue  # re-check without advancing i
            batch = texts[i : i + batch_size]
            try:
                results.extend(self._embed_batch(batch))
                i += batch_size
            except Exception as e:
                err = str(e)
                if ("memory" in err.lower() or "onnxruntime" in err.lower()) and batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    logger.warning(f"GPU OOM/error: reducing embed batch to {batch_size} and retrying")
                elif batch_size == 1 and self.backend != "CPU":
                    # GPU provider failed even at batch=1 — fall back to CPU session
                    import onnxruntime as ort
                    logger.warning(f"{self.backend} failed at batch=1 — falling back to CPU embeddings")
                    sess_opts = ort.SessionOptions()
                    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    cpu_threads = max(1, (os.cpu_count() or 2) // 2)
                    sess_opts.intra_op_num_threads = cpu_threads
                    self._session = ort.InferenceSession(
                        self._onnx_path,
                        sess_options=sess_opts,
                        providers=["CPUExecutionProvider"],
                    )
                    self.backend = "CPU (fallback)"
                    results.extend(self._embed_batch(batch))
                    i += batch_size
                else:
                    raise
        return results

    def embed_query(self, text: str) -> List[float]:
        cached = _embed_cache.get(text)
        if cached is not None:
            return cached
        # Signal background workers to pause their batches
        self._chat_priority.set()
        try:
            result = self._embed_batch([text])[0]
        finally:
            self._chat_priority.clear()
        _embed_cache.put(text, result)
        return result

    # ── Internal ───────────────────────────────────────────────────────────────

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        if self._batch_ok:
            return self._run_ort(texts)
        return [self._run_ort([t])[0] for t in texts]

    def _run_ort(self, texts: List[str]) -> List[List[float]]:
        enc = self._tokenizer(
            texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )
        ort_inputs = {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }

        if self._pre_pooled:
            vecs = self._session.run([self._out_name], ort_inputs)[0]
            return vecs.tolist()

        hidden = self._session.run([self._out_name], ort_inputs)[0]
        mask   = enc["attention_mask"][..., np.newaxis].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1e-9)
        norms  = np.linalg.norm(pooled, axis=-1, keepdims=True)
        return (pooled / norms.clip(min=1e-9)).tolist()


def try_build_native_embeddings() -> "NativeEmbeddings | None":
    """
    Returns a NativeEmbeddings instance (CUDA or CPU) if the ONNX model is available,
    otherwise returns None so the caller can fall back to HuggingFace.
    """
    try:
        return NativeEmbeddings()
    except FileNotFoundError:
        logger.info("ONNX embedding model not found — using HuggingFace CPU")
        return None
    except Exception as e:
        logger.warning(f"NativeEmbeddings failed to load: {e} — using HuggingFace CPU")
        return None
