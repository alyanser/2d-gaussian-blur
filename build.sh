#!/usr/bin/bash

# build tga library
cd tga
cmake -B build
cmake --build build -j$(nproc)

cd ..
clang++ convolution.cu -o convolution -L/opt/cuda/lib64 -lcudart_static -ldl -lrt -pthread -Itga -std=c++17 -Ltga/build -ltga-lib -O3
