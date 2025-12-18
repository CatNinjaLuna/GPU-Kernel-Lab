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
    """Plot GEMM performance comparing naive vs tiled."""
    df = pd.read_csv(csv_file)
    
    # Separate naive and tiled results
    naive_df = df[df['variant'] == 'naive'].copy()
    tiled_df = df[df['variant'] == 'tiled'].copy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # GFLOP/s comparison
    if not naive_df.empty:
        ax1.plot(naive_df['size'], naive_df['gflops'], 'o-', linewidth=2, markersize=8, 
                label='Naive', color='coral')
    if not tiled_df.empty:
        ax1.plot(tiled_df['size'], tiled_df['gflops'], 's-', linewidth=2, markersize=8, 
                label='Tiled (16×16)', color='green')
    
    ax1.set_xlabel('Matrix Size (N×N)')
    ax1.set_ylabel('Performance (GFLOP/s)')
    ax1.set_title('GEMM: Naive vs Tiled Performance')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.legend()
    
    # Speedup plot
    if not naive_df.empty and not tiled_df.empty:
        # Merge on size to calculate speedup
        merged = naive_df.merge(tiled_df, on='size', suffixes=('_naive', '_tiled'))
        speedup = merged['time_ms_naive'] / merged['time_ms_tiled']
        
        ax2.plot(merged['size'], speedup, 'd-', linewidth=2, markersize=8, color='purple')
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No speedup')
        ax2.set_xlabel('Matrix Size (N×N)')
        ax2.set_ylabel('Speedup (Naive/Tiled)')
        ax2.set_title('GEMM: Tiled Speedup Over Naive')
        ax2.set_xscale('log', base=2)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Annotate speedup values
        for _, row in merged.iterrows():
            sp = row['time_ms_naive'] / row['time_ms_tiled']
            ax2.annotate(f"{sp:.2f}×", 
                       xy=(row['size'], sp),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=8)
    
    plt.tight_layout()
    output_file = output_dir / 'gemm_naive_vs_tiled.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_reduction_performance(csv_file, output_dir):
    """Plot reduction optimization progression."""
    df = pd.read_csv(csv_file)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bandwidth comparison
    variants = df['variant'].tolist()
    bandwidths = df['bandwidth_gbs'].tolist()
    colors = ['coral', 'steelblue', 'green']
    
    bars = ax1.bar(range(len(variants)), bandwidths, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(variants)))
    ax1.set_xticklabels(['Atomic\n(Baseline)', 'Shared Memory\n(Tree Reduction)', 
                         'Warp Shuffle\n(Optimized)'])
    ax1.set_ylabel('Bandwidth (GB/s)')
    ax1.set_title('Reduction: Optimization Progression')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, bw in zip(bars, bandwidths):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{bw:.1f} GB/s',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Speedup comparison (relative to atomic)
    baseline_bw = df[df['variant'] == 'atomic']['bandwidth_gbs'].values[0]
    speedups = df['bandwidth_gbs'] / baseline_bw
    
    bars2 = ax2.bar(range(len(variants)), speedups, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(variants)))
    ax2.set_xticklabels(['Atomic', 'Shared Memory', 'Warp Shuffle'])
    ax2.set_ylabel('Speedup over Atomic')
    ax2.set_title('Reduction: Speedup Analysis')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    
    # Add speedup labels
    for bar, sp in zip(bars2, speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{sp:.0f}×',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'reduction_optimization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_summary_table(results_dir, output_dir):
    """Create a summary table of all results."""
    vec_add_file = results_dir / 'vec_add_results.csv'
    gemm_file = results_dir / 'gemm_results.csv'
    reduction_file = results_dir / 'reduction_results.csv'
    
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
        naive_df = df[df['variant'] == 'naive']
        tiled_df = df[df['variant'] == 'tiled']
        
        summary_lines.append(f"\n| Size | Variant | Time (ms) | Performance (GFLOP/s) | Status |\n")
        summary_lines.append(f"|------|---------|-----------|----------------------|--------|\n")
        
        # Interleave naive and tiled for same size
        for size in sorted(df['size'].unique()):
            naive_row = naive_df[naive_df['size'] == size]
            tiled_row = tiled_df[tiled_df['size'] == size]
            
            if not naive_row.empty:
                row = naive_row.iloc[0]
                summary_lines.append(
                    f"| {size}×{size} | Naive | {row['time_ms']:.3f} | "
                    f"{row['gflops']:.2f} | {row['status']} |\n"
                )
            if not tiled_row.empty:
                row = tiled_row.iloc[0]
                summary_lines.append(
                    f"| {size}×{size} | Tiled | {row['time_ms']:.3f} | "
                    f"{row['gflops']:.2f} | {row['status']} |\n"
                )
    
    summary_lines.append(f"\n## Parallel Reduction (10M elements)\n")
    
    if reduction_file.exists():
        df = pd.read_csv(reduction_file)
        baseline_bw = df[df['variant'] == 'atomic']['bandwidth_gbs'].values[0]
        
        summary_lines.append(f"\n| Variant | Time (ms) | Bandwidth (GB/s) | Speedup | Status |\n")
        summary_lines.append(f"|---------|-----------|------------------|---------|--------|\n")
        for _, row in df.iterrows():
            speedup = row['bandwidth_gbs'] / baseline_bw
            variant_name = row['variant'].replace('_', ' ').title()
            summary_lines.append(
                f"| {variant_name} | {row['time_ms']:.3f} | "
                f"{row['bandwidth_gbs']:.2f} | {speedup:.0f}× | {row['status']} |\n"
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
    
    # Plot reduction
    reduction_csv = results_dir / 'reduction_results.csv'
    if reduction_csv.exists():
        plot_reduction_performance(reduction_csv, results_dir)
    
    # Create summary
    create_summary_table(results_dir, results_dir)
    
    print()
    print(f"All plots saved to: {results_dir}")


if __name__ == '__main__':
    main()
