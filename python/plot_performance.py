#!/usr/bin/env python3
"""
Plot performance metrics from benchmark results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import argparse


def setup_plotting_style():
    """Configure matplotlib style."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10


def plot_vec_add_bandwidth(csv_file, output_dir):
    """Plot vector addition bandwidth vs size."""
    df = pd.read_csv(csv_file)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bandwidth plot
    ax1.plot(df['size'] / 1e6, df['bandwidth_gbs'], 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Vector Size (Million Elements)')
    ax1.set_ylabel('Bandwidth (GB/s)')
    ax1.set_title('Vector Addition: Memory Bandwidth')
    ax1.grid(True, alpha=0.3)
    
    # Time plot
    ax2.plot(df['size'] / 1e6, df['time_ms'], 's-', linewidth=2, markersize=8, color='coral')
    ax2.set_xlabel('Vector Size (Million Elements)')
    ax2.set_ylabel('Execution Time (ms)')
    ax2.set_title('Vector Addition: Execution Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'vec_add_performance.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_gemm_performance(csv_file, output_dir):
    """Plot GEMM performance vs size."""
    df = pd.read_csv(csv_file)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # GFLOP/s plot
    ax1.plot(df['size'], df['gflops'], 'o-', linewidth=2, markersize=8, color='green')
    ax1.set_xlabel('Matrix Size (N×N)')
    ax1.set_ylabel('Performance (GFLOP/s)')
    ax1.set_title('GEMM (Tiled): Compute Performance')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # Time plot (log scale)
    ax2.plot(df['size'], df['time_ms'], 's-', linewidth=2, markersize=8, color='purple')
    ax2.set_xlabel('Matrix Size (N×N)')
    ax2.set_ylabel('Execution Time (ms)')
    ax2.set_title('GEMM (Tiled): Execution Time')
    ax2.set_yscale('log')
    ax2.set_xscale('log', base=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'gemm_performance.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_gemm_scaling(csv_file, output_dir):
    """Plot GEMM scaling efficiency."""
    df = pd.read_csv(csv_file)
    
    # Calculate FLOPs and efficiency
    df['total_flops'] = 2 * df['size']**3
    df['measured_flops'] = df['gflops'] * 1e9 * (df['time_ms'] / 1000)
    df['efficiency'] = (df['measured_flops'] / df['total_flops']) * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['size'], df['gflops'], 'o-', linewidth=2, markersize=8, label='Achieved')
    ax.set_xlabel('Matrix Size (N×N)')
    ax.set_ylabel('Performance (GFLOP/s)')
    ax.set_title('GEMM Performance Scaling')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add annotations
    for _, row in df.iterrows():
        ax.annotate(f"{row['gflops']:.0f}", 
                   xy=(row['size'], row['gflops']),
                   xytext=(0, 10), textcoords='offset points',
                   ha='center', fontsize=8)
    
    plt.tight_layout()
    output_file = output_dir / 'gemm_scaling.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_summary_table(results_dir, output_dir):
    """Create a summary table of all results."""
    vec_add_file = results_dir / 'vec_add_results.csv'
    gemm_file = results_dir / 'gemm_results.csv'
    
    summary_lines = []
    summary_lines.append("# GPU Kernel Performance Summary\n")
    summary_lines.append(f"\n## Vector Addition\n")
    
    if vec_add_file.exists():
        df = pd.read_csv(vec_add_file)
        summary_lines.append(f"\n| Size (M) | Time (ms) | Bandwidth (GB/s) | Status |\n")
        summary_lines.append(f"|----------|-----------|------------------|--------|\n")
        for _, row in df.iterrows():
            summary_lines.append(
                f"| {row['size']/1e6:.1f} | {row['time_ms']:.3f} | "
                f"{row['bandwidth_gbs']:.2f} | {row['status']} |\n"
            )
    
    summary_lines.append(f"\n## Matrix Multiplication (GEMM)\n")
    
    if gemm_file.exists():
        df = pd.read_csv(gemm_file)
        summary_lines.append(f"\n| Size | Time (ms) | Performance (GFLOP/s) | Status |\n")
        summary_lines.append(f"|------|-----------|----------------------|--------|\n")
        for _, row in df.iterrows():
            summary_lines.append(
                f"| {row['size']}×{row['size']} | {row['time_ms']:.3f} | "
                f"{row['gflops']:.2f} | {row['status']} |\n"
            )
    
    output_file = output_dir / 'summary.md'
    with open(output_file, 'w') as f:
        f.writelines(summary_lines)
    
    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Plot GPU kernel performance results')
    parser.add_argument('results_dir', type=Path, help='Directory containing benchmark results')
    args = parser.parse_args()
    
    results_dir = args.results_dir
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)
    
    setup_plotting_style()
    
    print("Generating performance plots...")
    print()
    
    # Plot vector addition
    vec_add_csv = results_dir / 'vec_add_results.csv'
    if vec_add_csv.exists():
        plot_vec_add_bandwidth(vec_add_csv, results_dir)
    
    # Plot GEMM
    gemm_csv = results_dir / 'gemm_results.csv'
    if gemm_csv.exists():
        plot_gemm_performance(gemm_csv, results_dir)
        plot_gemm_scaling(gemm_csv, results_dir)
    
    # Create summary
    create_summary_table(results_dir, results_dir)
    
    print()
    print(f"All plots saved to: {results_dir}")


if __name__ == '__main__':
    main()
