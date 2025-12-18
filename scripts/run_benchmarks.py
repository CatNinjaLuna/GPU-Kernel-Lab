#!/usr/bin/env python3
"""
Benchmark runner for GPU kernels.
Runs tests with multiple sizes and saves results to CSV.
"""

import subprocess
import csv
import os
import re
from pathlib import Path
from datetime import datetime


def run_test(executable, args=None):
    """Run a test executable and capture output."""
    cmd = [executable]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running {executable}: {e}")
        return None


def parse_vec_add_output(output):
    """Parse vector addition test output."""
    time_match = re.search(r'GPU Time:\s+([\d.]+)\s+ms', output)
    bandwidth_match = re.search(r'Bandwidth:\s+([\d.]+)\s+GB/s', output)
    status_match = re.search(r'Result:\s+(\w+)', output)
    
    return {
        'time_ms': float(time_match.group(1)) if time_match else None,
        'bandwidth_gbs': float(bandwidth_match.group(1)) if bandwidth_match else None,
        'status': status_match.group(1) if status_match else None
    }


def parse_gemm_output(output):
    """Parse GEMM test output."""
    time_match = re.search(r'GPU Time:\s+([\d.]+)\s+ms', output)
    gflops_match = re.search(r'Performance:\s+([\d.]+)\s+GFLOP/s', output)
    status_match = re.search(r'Result:\s+(\w+)', output)
    
    return {
        'time_ms': float(time_match.group(1)) if time_match else None,
        'gflops': float(gflops_match.group(1)) if gflops_match else None,
        'status': status_match.group(1) if status_match else None
    }


def benchmark_vec_add(build_dir, output_dir):
    """Benchmark vector addition with various sizes."""
    sizes = [1000000, 5000000, 10000000, 20000000, 50000000]
    results = []
    
    test_exe = build_dir / 'tests' / 'test_vec_add.exe'
    if not test_exe.exists():
        test_exe = build_dir / 'tests' / 'test_vec_add'
    
    print("Benchmarking Vector Addition...")
    for size in sizes:
        print(f"  Size: {size:,}")
        output = run_test(str(test_exe), [str(size)])
        if output:
            data = parse_vec_add_output(output)
            data['size'] = size
            results.append(data)
    
    # Save to CSV
    csv_file = output_dir / 'vec_add_results.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['size', 'time_ms', 'bandwidth_gbs', 'status'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"  Results saved to {csv_file}")
    return results


def benchmark_gemm(build_dir, output_dir):
    """Benchmark GEMM with various sizes."""
    sizes = [256, 512, 1024, 2048, 4096]
    results = []
    
    test_exe = build_dir / 'tests' / 'test_gemm.exe'
    if not test_exe.exists():
        test_exe = build_dir / 'tests' / 'test_gemm'
    
    print("Benchmarking GEMM (Tiled)...")
    for size in sizes:
        print(f"  Size: {size}x{size}")
        output = run_test(str(test_exe), [str(size)])
        if output:
            data = parse_gemm_output(output)
            data['size'] = size
            data['variant'] = 'tiled'
            results.append(data)
    
    # Save to CSV
    csv_file = output_dir / 'gemm_results.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['size', 'variant', 'time_ms', 'gflops', 'status'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"  Results saved to {csv_file}")
    return results


def main():
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    build_dir = project_root / 'build'
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = project_root / 'results' / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"GPU Kernel Benchmarking")
    print(f"=" * 50)
    print(f"Output directory: {output_dir}")
    print()
    
    # Run benchmarks
    vec_add_results = benchmark_vec_add(build_dir, output_dir)
    print()
    gemm_results = benchmark_gemm(build_dir, output_dir)
    
    print()
    print(f"Benchmarking complete!")
    print(f"Results saved to: {output_dir}")
    print()
    print("Next steps:")
    print(f"  python python/plot_performance.py {output_dir}")


if __name__ == '__main__':
    main()
