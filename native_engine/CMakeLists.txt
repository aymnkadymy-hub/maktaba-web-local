cat << 'EOF' > native_engine/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(AymanComputeEngine)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)

# إضافة مسارات ONNX Runtime
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})

add_library(native_engine SHARED bindings/python_bindings.cpp src/document_optimizer.cpp src/universal_bridge.cpp)
target_link_libraries(native_engine PRIVATE pybind11::module ${OpenCV_LIBS} onnxruntime)
EOF