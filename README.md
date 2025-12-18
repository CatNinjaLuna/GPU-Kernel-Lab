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

-  **C++/CUDA kernels** with CMake build system
-  **CPU baseline** for correctness validation
-  **Performance benchmarking** with timing and throughput metrics
-  **Optimized implementations**:
   -  Grid-stride loops for vector operations
   -  Shared memory tiling for matrix multiplication
   -  Warp-level reductions for softmax
   -  Tiled convolution with halo loading
-  **Python environment** ready for analysis and visualization
-  **Optional**: Nsight Compute integration for advanced profiling

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

### Prerequisites

-  CUDA Toolkit 12.x or later
-  CMake 3.18+
-  C++ compiler with C++17 support
-  Python 3.10+ (for analysis scripts)
-  Conda or pip for Python dependencies

### 1. Setup Environment

```bash
# Create conda environment with dependencies
conda env create -f environment.yml
conda activate gpu-kernel-lab

# Or install with pip
pip install -r requirements.txt
```

### 2. Build

```bash
# On Windows with Visual Studio
& 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1' -Arch amd64
cd build
cmake .. -G Ninja -DCMAKE_CXX_COMPILER=cl -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe"
ninja

# On Linux
mkdir build && cd build
cmake ..
make -j
```

### 3. Run Tests

```bash
# From build directory
.\tests\test_vec_add.exe      # Windows
.\tests\test_gemm.exe
.\tests\test_softmax.exe
.\tests\test_conv2d.exe

# On Linux
./tests/test_vec_add
./tests/test_gemm
./tests/test_softmax
./tests/test_conv2d
```

### 4. View Results

Each test outputs:

-  Execution time (ms)
-  Performance metrics (GFLOP/s or GB/s)
-  Validation status (PASSED/FAILED)

### 5. Run Benchmarks and Generate Analysis

```bash
# Run comprehensive benchmarks across multiple sizes
### Vector Addition

| Size (M) | Time (ms) | Bandwidth (GB/s) | Status |
|----------|-----------|------------------|--------|
| 1.0      | 0.335     | 35.86            | PASSED |
| 5.0      | 0.467     | 128.38           | PASSED |
| 10.0     | 0.680     | 176.39           | PASSED |
| 20.0     | 0.749     | 320.28           | PASSED |
| 50.0     | 1.321     | 454.06           | PASSED |

### Matrix Multiplication (GEMM - Tiled)

| Size      | Time (ms) | Performance (GFLOP/s) | Status |
|-----------|-----------|----------------------|--------|
| 256×256   | 0.413     | 81.28                | PASSED |
| 512×512   | 0.486     | 552.86               | PASSED |
| 1024×1024 | 0.974     | 2205.93              | PASSED |
| 2048×2048 | 4.686     | 3666.25              | PASSED |
| 4096×4096 | 37.243    | 3690.29              | PASSED

# View generated plots and summary
# - vec_add_performance.png: Bandwidth and timing charts
# - gemm_performance.png: GFLOP/s and timing charts
# - gemm_scaling.png: Performance scaling analysis
# - summary.md: Performance summary table
```

---

## Example Output

| Kernel     | Size | Achieved GFLOP/s | % of Peak | Occupancy | Notes             |
| ---------- | ---- | ---------------- | --------- | --------- | ----------------- |
| vec_add    | 10M  | 750              | 82%       | 90%       | Memory-bound      |
| tiled_gemm | 4096 | 5200             | 65%       | 70%       | Tile size 128×128 |

Performance Analysis Tools

The project includes Python scripts for comprehensive performance analysis:

**Automated Benchmarking:**

-  `scripts/run_benchmarks.py` - Runs tests with multiple problem sizes
-  Automatically saves results to CSV with timestamps
-  Tests vector addition (1M to 50M elements) and GEMM (256×256 to 4096×4096)

**Visualization:**

-  `python/plot_performance.py` - Generates performance charts
   -  Bandwidth vs problem size for vector operations
   -  GFLOP/s vs matrix size for GEMM
   -  Performance scaling analysis
   -  Markdown summary tables

**Analysis:**

-  `python/compare_gemm.py` - Compare naive vs tiled implementations
-  Speedup calculations and visualizations

## Next Steps

-  ✅ ~~Create Python scripts for performance visualization~~
-  Add naive GEMM implementation for comparison
-  Parameter sweep automation (tile sizes, block sizes)
-  Integrate Nsight Compute for advanced profiling
-  Benchmark against cuBLAS/cuDNN baselines
-  Add PyTorch custom ops for GEMM benchmarking
-  Extend to WMMA/Tensor Core kernels (FP16/FP8)
-  Integrate Triton versions for ergonomic comparison
-  All kernels pass correctness tests against CPU reference
-  Performance benchmarks show expected speedups

## Next Steps

-  Create Python scripts for performance visualization
-  Add parameter sweep automation (tile sizes, block sizes)
-  Integrate Nsight Compute for advanced profiling
-  Benchmark against cuBLAS/cuDNN baselines
-  Add PyTorch custom ops for GEMM benchmarking
-  Extend to WMMA/Tensor Core kernels (FP16/FP8)
-  Integrate Triton versions for ergonomic comparison
-  Extend to WMMA/Tensor Core kernels (FP16/FP8)
-  Integrate Triton versions for ergonomic comparison
-  Automate parameter sweeps and performance reports
