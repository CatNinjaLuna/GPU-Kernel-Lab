# GPU Kernel Lab

A compact C++/CUDA playground for learning, validating, and profiling **core GPU kernels**.  
This project is designed to build intuition for GPU performance tuning—covering memory coalescing, shared memory tiling, and occupancy analysis.

---

## Overview

This lab implements and benchmarks several fundamental GPU kernels:

-  **Vector Addition (vec_add)** – baseline for memory coalescing and grid-stride loops
-  **Tiled Matrix Multiplication (GEMM)** – explores shared memory, tiling strategies, and performance scaling
-  **Softmax** – demonstrates warp-level reductions and numerical stability
-  **Convolution (conv2d)** – applies tiling and data reuse for 2D image processing

Each kernel is validated against a CPU reference implementation and profiled using **NVIDIA Nsight Compute**.

---

## Features

-  C++/CUDA kernels with CMake build system
-  CPU baseline for correctness checking
-  Profiling scripts that run Nsight Compute and export CSV metrics
-  Python utilities to:
   -  Parse Nsight Compute CSV logs
   -  Generate roofline plots
   -  Compare tuning configurations

---

## Project Structure

```
gpu-kernel-lab/
├─ src/             # CUDA kernel implementations
├─ tests/           # Unit tests for correctness and performance
├─ python/          # Nsight parsing and roofline plotting
├─ scripts/         # Profiling helpers and automation
├─ CMakeLists.txt
└─ README.md
```

---

## Getting Started

### 1. Build

```bash
mkdir build && cd build
cmake ..
make -j
```

### 2. Run Tests

```bash
./tests/test_vec_add
./tests/test_gemm
```

### 3. Profile a Kernel

```bash
ncu --target-processes all ./tests/test_gemm --size 1024
```

### 4. Generate Roofline Plot

```bash
python python/roofline.py runs/2025-10-15/gemm_tiled.csv
```

---

## Example Output

| Kernel     | Size | Achieved GFLOP/s | % of Peak | Occupancy | Notes             |
| ---------- | ---- | ---------------- | --------- | --------- | ----------------- |
| vec_add    | 10M  | 750              | 82%       | 90%       | Memory-bound      |
| tiled_gemm | 4096 | 5200             | 65%       | 70%       | Tile size 128×128 |

---

## Next Steps

-  Add PyTorch custom ops for GEMM benchmarking
-  Extend to WMMA/Tensor Core kernels (FP16/FP8)
-  Integrate Triton versions for ergonomic comparison
-  Automate parameter sweeps and performance reports
