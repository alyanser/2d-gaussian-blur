#!/usr/bin/bash

cuda_path="/opt/cuda"

# build tga library
cd tga
cmake -B build
cmake --build build -j$(nproc)

cd ..
clang++ convolution.cu -o convolution -L${cuda_path}/lib64 -lcudart_static -ldl -lrt -pthread -Itga -std=c++17 -Ltga/build -ltga-lib -O2 --cuda-path=${cuda_path}
