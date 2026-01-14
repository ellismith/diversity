#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
import glob

def load_all_alpha_diversity(diversity_dir, region, level='subtype'):
    """Load all alpha diversity files for a region"""
    if level == 'celltype':
        pattern = f"{region}_celltype_alpha_diversity.csv"
        files = [os.path.join(diversity_dir, pattern)]
    else:
        pattern = f"{region}_*_subtype_alpha_diversity.csv"
        files = glob.glob(os.path.join(diversity_dir, pattern))
    
    dfs = []
    for f in files:
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f, index_col=0)
        # Extract cell type name from filename
        basename = os.path.basename(f)
        if level == 'celltype':
            celltype = 'All Cell Types'
        else:
            # e.g., "HIP_GABAergic_neurons_subtype_alpha_diversity.csv"
            celltype = basename.replace(f"{region}_", "").replace("_subtype_alpha_diversity.csv", "")
            celltype = celltype.replace("_", " ").title()
        
        df['cell_type'] = celltype
        dfs.append(df)
    
    if len(dfs) == 0:
        return None
    
    return pd.concat(dfs, ignore_index=False)

def load_all_beta_diversity(diversity_dir, region, level='subtype'):
    """Load all beta diversity files and calculate mean distance per individual"""
    if level == 'celltype':
        pattern = f"{region}_celltype_beta_diversity.csv"
        files = [os.path.join(diversity_dir, pattern)]
    else:
        pattern = f"{region}_*_subtype_beta_diversity.csv"
        files = glob.glob(os.path.join(diversity_dir, pattern))
    
    dfs = []
    for f in files:
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f, index_col=0)
        
        # Calculate mean Bray-Curtis distance for each individual
        mean_distances = df.mean(axis=1)  # Mean distance from this individual to all others
        
        # Extract cell type name
        basename = os.path.basename(f)
        if level == 'celltype':
            celltype = 'All Cell Types'
        else:
            celltype = basename.replace(f"{region}_", "").replace("_subtype_beta_diversity.csv", "")
            celltype = celltype.replace("_", " ").title()
        
        temp_df = pd.DataFrame({
            'mean_bray_curtis': mean_distances,
            'cell_type': celltype
        })
        dfs.append(temp_df)
    
    if len(dfs) == 0:
        return None
    
    return pd.concat(dfs, ignore_index=False)

def plot_alpha_comparison(df, title, out_file):
    """Violin + strip plot of alpha diversity across cell types"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Violin plot
    sns.violinplot(data=df, x='cell_type', y='shannon_diversity', ax=ax, 
                   inner=None, color='lightgray', alpha=0.6)
    
    # Strip plot overlay
    sns.stripplot(data=df, x='cell_type', y='shannon_diversity', ax=ax,
                  size=5, alpha=0.6, jitter=True)
    
    ax.set_xlabel('Cell Type', fontsize=12)
    ax.set_ylabel('Shannon Diversity', fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file}")

def plot_beta_comparison(df, title, out_file):
    """Violin + strip plot of mean beta diversity across cell types"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Violin plot
    sns.violinplot(data=df, x='cell_type', y='mean_bray_curtis', ax=ax,
                   inner=None, color='lightgray', alpha=0.6)
    
    # Strip plot overlay
    sns.stripplot(data=df, x='cell_type', y='mean_bray_curtis', ax=ax,
                  size=5, alpha=0.6, jitter=True)
    
    ax.set_xlabel('Cell Type', fontsize=12)
    ax.set_ylabel('Mean Bray-Curtis Distance', fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Plot summary diversity comparisons")
    parser.add_argument("--region", type=str, required=True, help="Region name (e.g., HIP, CN)")
    parser.add_argument("--level", type=str, choices=['celltype', 'subtype'], default='subtype',
                        help="Compare cell types or subtypes")
    parser.add_argument("--diversity_dir", type=str, default="/scratch/easmit31/variability/diversity")
    parser.add_argument("--out_dir", type=str, default="/scratch/easmit31/variability/diversity_figs")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Loading {args.level} diversity data for {args.region}...")
    
    # Alpha diversity comparison
    alpha_df = load_all_alpha_diversity(args.diversity_dir, args.region, args.level)
    if alpha_df is not None:
        alpha_title = f"{args.region} Alpha Diversity Comparison ({args.level.title()})"
        alpha_out = os.path.join(args.out_dir, f"{args.region}_{args.level}_alpha_comparison.png")
        plot_alpha_comparison(alpha_df, alpha_title, alpha_out)
    else:
        print(f"No alpha diversity files found for {args.region}")
    
    # Beta diversity comparison
    beta_df = load_all_beta_diversity(args.diversity_dir, args.region, args.level)
    if beta_df is not None:
        beta_title = f"{args.region} Beta Diversity Comparison ({args.level.title()})"
        beta_out = os.path.join(args.out_dir, f"{args.region}_{args.level}_beta_comparison.png")
        plot_beta_comparison(beta_df, beta_title, beta_out)
    else:
        print(f"No beta diversity files found for {args.region}")

if __name__ == "__main__":
    main()
