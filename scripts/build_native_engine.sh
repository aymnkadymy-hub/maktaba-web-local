#!/bin/bash
echo "🚀 Starting Build for Ayman..."
cd native_engine && mkdir -p build && cd build
cmake ..
make -j$(nproc)
if [ -f "native_engine.so" ]; then
    echo "✅ Success!"
    cp native_engine.so ../../backend/utils/
else
    echo "❌ Failed!"
fi
