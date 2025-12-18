# Python Analysis Scripts

This directory contains Python scripts for analyzing and visualizing GPU kernel performance.

## Scripts

### `plot_performance.py`

Generates performance plots from benchmark results.

```bash
# After running benchmarks
python python/plot_performance.py results/20251218_120000/
```

Generates:

-  `vec_add_performance.png` - Bandwidth and time plots for vector addition
-  `gemm_performance.png` - GFLOP/s and time plots for GEMM
-  `gemm_scaling.png` - Performance scaling analysis
-  `summary.md` - Markdown summary table

### `compare_gemm.py`

Compare naive vs tiled GEMM implementations.

```bash
python python/compare_gemm.py results/naive_gemm.csv results/tiled_gemm.csv
```

Generates:

-  `gemm_comparison.png` - Side-by-side performance and speedup comparison

## Dependencies

All dependencies are included in `requirements.txt`:

-  numpy
-  pandas
-  matplotlib
-  seaborn

Install with:

```bash
pip install -r ../requirements.txt
```

## Workflow

1. **Run benchmarks:**

   ```bash
   python scripts/run_benchmarks.py
   ```

2. **Generate plots:**

   ```bash
   python python/plot_performance.py results/<timestamp>/
   ```

3. **View results:**
   -  Check PNG files in results directory
   -  Read `summary.md` for tabular data
