#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from glob import glob

def plot_entropy_bars_celltype(df, cell_type, region_name, out_file):
    """Plot bar graph for a specific cell type's cluster diversity"""
    df_plot = df[df['cell_type'] == cell_type].copy()
    df_plot = df_plot.sort_values('age')
    
    fig, ax = plt.subplots(figsize=(16, 6))
    x_pos = np.arange(len(df_plot))
    bars = ax.bar(x_pos, df_plot['adjusted_entropy'], color='steelblue', alpha=0.7, width=0.8)
    
    labels = [f"{aid}, {age:.1f}" for aid, age in zip(df_plot['animal_id'], df_plot['age'])]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=7)
    
    ax.set_ylabel("Adjusted Entropy", fontsize=12)
    ax.set_xlabel("Individual", fontsize=12)
    ax.set_ylim(-1.0, 0.0)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=-1, color='red', linestyle='--', alpha=0.3, linewidth=1)
    plt.title(f"{cell_type} cluster diversity - {region_name}\n(0 = max diversity, -1 = min diversity)", fontsize=13)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_file}")

def plot_entropy_scatter_celltype(df, cell_type, region_name, out_file):
    """Plot scatter of age vs cell-type-level entropy"""
    df_plot = df[df['cell_type'] == cell_type].copy()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df_plot['age'], df_plot['adjusted_entropy'], alpha=0.6, s=50, color='steelblue')
    
    # Fit trend line
    z = np.polyfit(df_plot['age'], df_plot['adjusted_entropy'], 1)
    p = np.poly1d(z)
    
    # Create smooth x values for trend line
    x_smooth = np.linspace(df_plot['age'].min(), df_plot['age'].max(), 100)
    y_smooth = p(x_smooth)
    
    # Plot trend line as continuous line
    ax.plot(x_smooth, y_smooth, "r-", alpha=0.8, linewidth=2, linestyle='--')
    
    corr = np.corrcoef(df_plot['age'], df_plot['adjusted_entropy'])[0, 1]
    
    ax.set_xlabel("Age", fontsize=11)
    ax.set_ylabel("Adjusted Entropy", fontsize=11)
    ax.set_ylim(-1.0, 0.0)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=-1, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    plt.title(f"{cell_type} cluster diversity vs age - {region_name}\n(0 = max diversity, -1 = min diversity)\nPearson r = {corr:.3f}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Plot entropy metrics for each cell type separately")
    parser.add_argument("--entropy_dir", type=str, default="/scratch/easmit31/variability/entropy_csvs",
                        help="Directory with entropy CSV files")
    parser.add_argument("--out_dir", type=str, default="/scratch/easmit31/variability/entropy_figs_separate",
                        help="Output directory for figures")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Find all entropy CSV files
    csv_files = glob(os.path.join(args.entropy_dir, "entropy_*.csv"))
    
    for csv_file in csv_files:
        region_name = os.path.basename(csv_file).replace("entropy_", "").replace(".csv", "")
        print(f"\nProcessing {region_name}...")
        
        df = pd.read_csv(csv_file)
        
        # Get all cell types except 'all' (which is region-level)
        cell_types = [ct for ct in df['cell_type'].unique() if ct != 'all']
        
        for cell_type in cell_types:
            print(f"  Plotting {cell_type}...")
            
            # Bar plot
            out_bar = os.path.join(args.out_dir, f"entropy_bar_{cell_type}_{region_name}.png")
            plot_entropy_bars_celltype(df, cell_type, region_name, out_bar)
            
            # Scatter plot
            out_scatter = os.path.join(args.out_dir, f"entropy_scatter_{cell_type}_{region_name}.png")
            plot_entropy_scatter_celltype(df, cell_type, region_name, out_scatter)

if __name__ == "__main__":
    main()
