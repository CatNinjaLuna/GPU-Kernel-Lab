#!/usr/bin/env python3
"""
Compare naive vs tiled GEMM implementations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def setup_plotting_style():
    """Configure matplotlib style."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10


def plot_gemm_comparison(naive_csv, tiled_csv, output_dir):
    """Compare naive and tiled GEMM performance."""
    df_naive = pd.read_csv(naive_csv)
    df_tiled = pd.read_csv(tiled_csv)
    
    df_naive['variant'] = 'Naive'
    df_tiled['variant'] = 'Tiled'
    df_combined = pd.concat([df_naive, df_tiled])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Performance comparison
    for variant in ['Naive', 'Tiled']:
        df_var = df_combined[df_combined['variant'] == variant]
        ax1.plot(df_var['size'], df_var['gflops'], 'o-', 
                linewidth=2, markersize=8, label=variant)
    
    ax1.set_xlabel('Matrix Size (N×N)')
    ax1.set_ylabel('Performance (GFLOP/s)')
    ax1.set_title('GEMM: Naive vs Tiled Performance')
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Speedup
    df_speedup = pd.merge(
        df_naive[['size', 'gflops']].rename(columns={'gflops': 'naive_gflops'}),
        df_tiled[['size', 'gflops']].rename(columns={'gflops': 'tiled_gflops'}),
        on='size'
    )
    df_speedup['speedup'] = df_speedup['tiled_gflops'] / df_speedup['naive_gflops']
    
    ax2.bar(range(len(df_speedup)), df_speedup['speedup'], 
           color='steelblue', edgecolor='black')
    ax2.set_xlabel('Matrix Size (N×N)')
    ax2.set_ylabel('Speedup (Tiled / Naive)')
    ax2.set_title('Tiled GEMM Speedup over Naive')
    ax2.set_xticks(range(len(df_speedup)))
    ax2.set_xticklabels(df_speedup['size'])
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    # Add value labels on bars
    for i, v in enumerate(df_speedup['speedup']):
        ax2.text(i, v + 0.1, f'{v:.2f}×', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_file = output_dir / 'gemm_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # Print summary
    print("\nSpeedup Summary:")
    print(df_speedup.to_string(index=False))
    
    return df_speedup


def main():
    parser = argparse.ArgumentParser(description='Compare GEMM variants')
    parser.add_argument('naive_csv', type=Path, help='CSV file with naive GEMM results')
    parser.add_argument('tiled_csv', type=Path, help='CSV file with tiled GEMM results')
    parser.add_argument('--output-dir', type=Path, default=None, 
                       help='Output directory for plots')
    args = parser.parse_args()
    
    if not args.naive_csv.exists():
        print(f"Error: File not found: {args.naive_csv}")
        return
    
    if not args.tiled_csv.exists():
        print(f"Error: File not found: {args.tiled_csv}")
        return
    
    output_dir = args.output_dir or args.tiled_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_plotting_style()
    
    print("Comparing GEMM implementations...")
    plot_gemm_comparison(args.naive_csv, args.tiled_csv, output_dir)
    print(f"\nComparison plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
