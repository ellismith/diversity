#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from glob import glob

def plot_entropy_bars_region(df, region_name, out_file):
    """Plot bar graph of region-level entropy (cell type diversity)"""
    df_plot = df[df['cell_type'] == 'all'].copy()
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
    plt.title(f"Region-level cell type diversity - {region_name}\n(0 = max diversity, -1 = min diversity)", fontsize=13)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_file}")

def plot_entropy_scatter_region(df, region_name, out_file):
    """Plot scatter of age vs region-level entropy"""
    df_plot = df[df['cell_type'] == 'all'].copy()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df_plot['age'], df_plot['adjusted_entropy'], alpha=0.6, s=50, color='steelblue')
    
    z = np.polyfit(df_plot['age'], df_plot['adjusted_entropy'], 1)
    p = np.poly1d(z)
    ax.plot(df_plot['age'], p(df_plot['age']), "r--", alpha=0.8, linewidth=2)
    
    corr = np.corrcoef(df_plot['age'], df_plot['adjusted_entropy'])[0, 1]
    
    ax.set_xlabel("Age", fontsize=11)
    ax.set_ylabel("Adjusted Entropy", fontsize=11)
    ax.set_ylim(-1.0, 0.0)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=-1, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    plt.title(f"Region-level cell type diversity vs age - {region_name}\n(0 = max diversity, -1 = min diversity)\nPearson r = {corr:.3f}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_file}")

def plot_entropy_bars_celltype(df, cell_type, region_name, out_file):
    """Plot bar graph for a specific cell type's cluster diversity"""
    df_plot = df[df['cell_type'] == cell_type].copy()
    df_plot = df_plot.sort_values('age')
    
    fig, ax = plt.subplots(figsize=(16, 6))
    x_pos = np.arange(len(df_plot))
    bars = ax.bar(x_pos, df_plot['adjusted_entropy'], color='coral', alpha=0.7, width=0.8)
    
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
    ax.scatter(df_plot['age'], df_plot['adjusted_entropy'], alpha=0.6, s=50, color='coral')
    
    z = np.polyfit(df_plot['age'], df_plot['adjusted_entropy'], 1)
    p = np.poly1d(z)
    ax.plot(df_plot['age'], p(df_plot['age']), "r--", alpha=0.8, linewidth=2)
    
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

def plot_entropy_bars_all_celltypes(df, region_name, out_file):
    """Plot grouped bar graph with all cell types side-by-side"""
    df_plot = df[df['cell_type'] != 'all'].copy()
    
    cell_types = sorted(df_plot['cell_type'].unique())
    
    # Use a more distinct color palette
    # Manually assign colors that are visually distinct
    distinct_colors = [
        '#e41a1c',  # red
        '#377eb8',  # blue
        '#4daf4a',  # green
        '#984ea3',  # purple
        '#ff7f00',  # orange
        '#ffff33',  # yellow
        '#a65628',  # brown
        '#f781bf',  # pink
        '#999999',  # gray
        '#1f77b4',  # medium blue
    ]
    
    # If more than 10 cell types, use tab20
    if len(cell_types) > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, len(cell_types)))
    else:
        colors = distinct_colors[:len(cell_types)]
    
    color_map = dict(zip(cell_types, colors))
    
    fig, ax = plt.subplots(figsize=(18, 6))
    
    individuals = df_plot[['animal_id', 'age']].drop_duplicates().sort_values('age')
    x_pos = np.arange(len(individuals))
    
    bar_width = 0.8 / len(cell_types)
    for i, ct in enumerate(cell_types):
        df_ct = df_plot[df_plot['cell_type'] == ct].set_index('animal_id')
        heights = [df_ct.loc[aid, 'adjusted_entropy'] if aid in df_ct.index else 0 
                   for aid in individuals['animal_id']]
        offset = (i - len(cell_types)/2 + 0.5) * bar_width
        ax.bar(x_pos + offset, heights, bar_width, label=ct, color=color_map[ct], alpha=0.9)
    
    labels = [f"{aid}, {age:.1f}" for aid, age in zip(individuals['animal_id'], individuals['age'])]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=6)
    
    ax.set_ylabel("Adjusted Entropy", fontsize=12)
    ax.set_xlabel("Individual", fontsize=12)
    ax.set_ylim(-1.0, 0.0)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=-1, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=1)
    plt.title(f"Cluster diversity by cell type - {region_name}\n(0 = max diversity, -1 = min diversity)", fontsize=13)
    plt.subplots_adjust(bottom=0.15, right=0.88)
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Plot entropy metrics")
    parser.add_argument("--entropy_dir", type=str, default="/scratch/easmit31/variability/entropy_csvs",
                        help="Directory with entropy CSV files")
    parser.add_argument("--out_dir", type=str, default="/scratch/easmit31/variability/entropy_figs",
                        help="Directory to save figures")
    parser.add_argument("--mode", type=str, default="within_region", 
                        choices=["within_region", "within_cell_type"],
                        help="Mode used to calculate entropy")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    csv_files = glob(os.path.join(args.entropy_dir, "entropy_*.csv"))
    
    for csv_file in csv_files:
        region_name = os.path.basename(csv_file).replace("entropy_", "").replace(".csv", "")
        print(f"\nProcessing {region_name}...")
        
        df = pd.read_csv(csv_file)
        
        if args.mode == "within_region":
            out_bar = os.path.join(args.out_dir, f"entropy_bar_region_{region_name}.png")
            out_scatter = os.path.join(args.out_dir, f"entropy_scatter_region_{region_name}.png")
            
            plot_entropy_bars_region(df, region_name, out_bar)
            plot_entropy_scatter_region(df, region_name, out_scatter)
            
        elif args.mode == "within_cell_type":
            out_bar_combined = os.path.join(args.out_dir, f"entropy_bar_all_celltypes_{region_name}.png")
            plot_entropy_bars_all_celltypes(df, region_name, out_bar_combined)
            
            cell_types = [ct for ct in df['cell_type'].unique() if ct != 'all']
            for cell_type in cell_types:
                out_bar = os.path.join(args.out_dir, f"entropy_bar_{cell_type}_{region_name}.png")
                out_scatter = os.path.join(args.out_dir, f"entropy_scatter_{cell_type}_{region_name}.png")
                
                plot_entropy_bars_celltype(df, cell_type, region_name, out_bar)
                plot_entropy_scatter_celltype(df, cell_type, region_name, out_scatter)

if __name__ == "__main__":
    main()
