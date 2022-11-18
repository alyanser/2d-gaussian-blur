#!/usr/bin/bash

# build tga library
cd tga
cmake -B build
cmake --build build

cd ..
nvcc convolution.cu -Itga -std=c++17 -Ltga/build -ltga-lib -o convolution
