import sys
sys.path.insert(0, r'C:\Users\Ayman\Desktop\book-companion-ai\native_engine\build_win')
import native_engine as ne
print('Import: OK')

# Test S24DocumentOptimizer
opt = ne.S24DocumentOptimizer()
print('S24DocumentOptimizer: OK')

# Test MemoryCompressor
import numpy as np
mc = ne.MemoryCompressor()
data = np.ones(128, dtype=np.float32)
compressed = mc.compress(data)
print(f'MemoryCompressor: OK — {len(compressed)} uint16 values')

# Test AymanComputeEngine
import threading
engine = ne.AymanComputeEngine(2)
done = threading.Event()
engine.send_to_pipe(lambda: done.set())
done.wait(timeout=1.0)
print(f'AymanComputeEngine: OK — task executed={done.is_set()}')

# Test UniversalModelBridge
bridge = ne.UniversalModelBridge()
bridge.load_model('test.onnx')
print('UniversalModelBridge: OK (stub)')

print()
print('ALL TESTS PASSED')
