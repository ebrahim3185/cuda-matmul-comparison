# Performance Comparison of CPU and CUDA Matrix Multiplication  
### A Study of Naive vs. Tiled Shared-Memory Implementations


---

## Overview

This project compares the performance of three matrix-multiplication implementations for square matrices of size \(N = 1024\):

1. **Single-core CPU** – straightforward triple-loop baseline  
2. **Naive CUDA** – one thread per output element, pure global-memory access  
3. **Tiled CUDA** – classic shared-memory tiling optimization

Experiments were run on an **NVIDIA GeForce RTX 3050 Laptop GPU**.  
Results show that the naïve CUDA kernel already delivers ~**670×** speedup over the single-core CPU version, while the tiled shared-memory version adds another **~31 %**, reaching roughly **880×** overall speedup.

The study highlights the decisive impact of the GPU memory hierarchy and cooperative use of shared memory in reducing global-memory traffic.

**Keywords:** CUDA, Matrix Multiplication, Shared Memory, Tiling, Parallel Computing, Performance Optimization


---

## Requirements

- C++ compiler (`g++` or `clang++`)
- NVIDIA CUDA Toolkit (`nvcc`)
- NVIDIA GPU with CUDA support (tested on RTX 3050 Laptop)

---

## Compilation

### CPU Single Core version
g++ -O3 -o cpu_matmul cpu_matmul.cpp

### GPU Naive CUDA version

g++ -O3 -o cpu_matmul cpu_matmul.cpp

### GPU Tiled CUDA version

g++ -O3 -o cpu_matmul cpu_matmul.cpp

### Execution
- ./cpu_matmul
- ./naive_matmul
- ./tiled_matmul

**Author:** Ebrahim Khodadadi  
**Course:** Operating Systems  
**Date:** October 2026  
**Email:** khodadadie56@gmail.com  







