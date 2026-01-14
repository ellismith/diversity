#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from sklearn.manifold import MDS

def plot_alpha_scatter(alpha_df, title, out_file):
    """Scatter plot of Shannon diversity vs age"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(alpha_df['age'], alpha_df['shannon_diversity'], 
                        c=alpha_df['age'], cmap='viridis', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Age (years)', fontsize=13)
    ax.set_ylabel('Shannon Diversity', fontsize=13)
    ax.set_title(title, fontsize=14, pad=15)
    ax.grid(alpha=0.3, ls='--')
    
    plt.colorbar(scatter, ax=ax, label='Age (years)')
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file}")

def plot_beta_mds(beta_df, alpha_df, title, out_file):
    """MDS plot of beta diversity colored by age"""
    # Run MDS on distance matrix
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(beta_df.values)
    
    # Match ages to individuals
    ages = alpha_df.loc[beta_df.index, 'age'].values
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                        c=ages, cmap='viridis', s=150, alpha=0.7, 
                        edgecolors='black', linewidth=0.5)
    
    # Optionally label points with animal IDs
    for i, animal_id in enumerate(beta_df.index):
        ax.annotate(animal_id, (coords[i, 0], coords[i, 1]), 
                   fontsize=7, alpha=0.6, ha='center', va='bottom')
    
    ax.set_xlabel('MDS1', fontsize=13)
    ax.set_ylabel('MDS2', fontsize=13)
    ax.set_title(title, fontsize=14, pad=15)
    ax.grid(alpha=0.3, ls='--')
    
    plt.colorbar(scatter, ax=ax, label='Age (years)')
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Plot diversity metrics")
    parser.add_argument("--alpha_file", type=str, required=True,
                        help="Alpha diversity CSV file (must include 'age' column)")
    parser.add_argument("--beta_file", type=str, required=True,
                        help="Beta diversity CSV file")
    parser.add_argument("--out_dir", type=str, 
                        default="/scratch/easmit31/variability/diversity_figs")
    parser.add_argument("--title_prefix", type=str, default="",
                        help="Prefix for plot titles (e.g., 'HIP GABAergic')")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load data
    alpha_df = pd.read_csv(args.alpha_file, index_col=0)
    beta_df = pd.read_csv(args.beta_file, index_col=0)
    
    # Check that age column exists
    if 'age' not in alpha_df.columns:
        print(f"ERROR: 'age' column not found in {args.alpha_file}")
        print(f"Available columns: {alpha_df.columns.tolist()}")
        return
    
    # Generate output filenames from input
    base_name = os.path.basename(args.alpha_file).replace('_alpha_diversity.csv', '')
    
    # Plot alpha diversity
    alpha_title = f"{args.title_prefix} Alpha Diversity" if args.title_prefix else "Alpha Diversity"
    alpha_out = os.path.join(args.out_dir, f"{base_name}_alpha_scatter.png")
    plot_alpha_scatter(alpha_df, alpha_title, alpha_out)
    
    # Plot beta diversity MDS
    beta_title = f"{args.title_prefix} Beta Diversity (MDS)" if args.title_prefix else "Beta Diversity (MDS)"
    beta_out = os.path.join(args.out_dir, f"{base_name}_beta_mds.png")
    plot_beta_mds(beta_df, alpha_df, beta_title, beta_out)

if __name__ == "__main__":
    main()
