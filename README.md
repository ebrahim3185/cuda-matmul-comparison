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


## Compilation Environment

All CUDA codes in this project **must be compiled** using the **x64 Native Tools Command Prompt for VS** (Visual Studio Developer Command Prompt).

### How to open it:
1. Search for **"x64 Native Tools Command Prompt for VS"** in the Windows Start Menu  
2. Or go to:  
   `Start → Visual Studio 2026 → x64 Native Tools Command Prompt for VS 2026`

> **Important:**  
> Do **not** use the regular Command Prompt or PowerShell.  
> The CUDA compiler (`nvcc`) and the required environment variables are only available inside the **x64 Native Tools Command Prompt**.
### CPU Single Core version
cl /O2 /EHsc src\cpu_matmul.cpp /Fe:cpu_matmul.exe

### GPU Naive CUDA version

nvcc -O3 src\naive_matmul.cu -o naive_matmul.exe

### GPU Tiled CUDA version

nvcc -O3 src\tiled_matmul.cu -o tiled_matmul.exe

### Execution
- cpu_matmul.exe  
- naive_matmul.exe  
- tiled_matmul.exe  


**Author:** Ebrahim Khodadadi  
**Course:** Operating Systems  
**Date:** October 2026  
**Email:** khodadadie56@gmail.com  







