#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from sklearn.manifold import MDS

def plot_alpha_scatter(alpha_df, metric_name, title, out_file):
    """Scatter plot of alpha diversity vs age"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(alpha_df['age'], alpha_df[metric_name], 
                        c=alpha_df['age'], cmap='viridis', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Age (years)', fontsize=13)
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=13)
    ax.set_title(title, fontsize=14, pad=15)
    ax.grid(alpha=0.3, ls='--')
    
    plt.colorbar(scatter, ax=ax, label='Age (years)')
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file}")

def plot_beta_mds(beta_df, alpha_df, title, out_file):
    """MDS plot of beta diversity colored by age - handles missing data"""
    
    # Check for missing data (NaN or individuals not in alpha_df)
    valid_individuals = [ind for ind in beta_df.index if ind in alpha_df.index]
    
    if len(valid_individuals) < 3:
        print(f"Warning: Only {len(valid_individuals)} individuals available for MDS. Skipping plot.")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f'Insufficient data for MDS\n({len(valid_individuals)} individuals)', 
                ha='center', va='center', fontsize=14)
        ax.set_title(title, fontsize=14, pad=15)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved placeholder: {out_file}")
        return
    
    # Filter beta matrix to valid individuals
    beta_filtered = beta_df.loc[valid_individuals, valid_individuals]
    
    # Iteratively remove individuals with too many NaN values
    max_iterations = 20
    for iteration in range(max_iterations):
        if not beta_filtered.isna().any().any():
            break
        
        # Count NaN per individual
        nan_counts = beta_filtered.isna().sum(axis=1)
        
        # Find individual with most NaN
        worst_individual = nan_counts.idxmax()
        worst_nan_count = nan_counts[worst_individual]
        
        if worst_nan_count == 0:
            break
        
        print(f"  Removing {worst_individual} with {worst_nan_count} NaN values")
        
        # Remove this individual
        valid_individuals.remove(worst_individual)
        beta_filtered = beta_filtered.drop(index=worst_individual, columns=worst_individual)
        
        if len(valid_individuals) < 3:
            break
    
    if len(valid_individuals) < 3:
        print(f"Warning: Only {len(valid_individuals)} individuals after removing NaN. Skipping MDS.")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f'Insufficient data for MDS\nafter removing missing values\n({len(valid_individuals)} valid individuals)', 
                ha='center', va='center', fontsize=14)
        ax.set_title(title, fontsize=14, pad=15)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved placeholder: {out_file}")
        return
    
    # Run MDS on filtered distance matrix
    try:
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coords = mds.fit_transform(beta_filtered.values)
    except Exception as e:
        print(f"Error running MDS: {e}")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f'MDS failed:\n{str(e)}', ha='center', va='center', fontsize=12)
        ax.set_title(title, fontsize=14, pad=15)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved error placeholder: {out_file}")
        return
    
    # Match ages to filtered individuals
    ages = alpha_df.loc[valid_individuals, 'age'].values
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                        c=ages, cmap='viridis', s=150, alpha=0.7, 
                        edgecolors='black', linewidth=0.5)
    
    # Optionally label points with animal IDs
    for i, animal_id in enumerate(valid_individuals):
        ax.annotate(animal_id, (coords[i, 0], coords[i, 1]), 
                   fontsize=7, alpha=0.6, ha='center', va='bottom')
    
    ax.set_xlabel('MDS1', fontsize=13)
    ax.set_ylabel('MDS2', fontsize=13)
    ax.set_title(title, fontsize=14, pad=15)
    ax.grid(alpha=0.3, ls='--')
    
    plt.colorbar(scatter, ax=ax, label='Age (years)')
    
    n_excluded = len(beta_df) - len(valid_individuals)
    if n_excluded > 0:
        ax.text(0.02, 0.98, f'Note: {n_excluded} individuals excluded due to missing data',
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
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
    
    # Detect which metric (shannon_diversity or adjusted_entropy)
    if 'shannon_diversity' in alpha_df.columns:
        metric_name = 'shannon_diversity'
    elif 'adjusted_entropy' in alpha_df.columns:
        metric_name = 'adjusted_entropy'
    else:
        print(f"ERROR: No recognized diversity metric in {args.alpha_file}")
        print(f"Available columns: {alpha_df.columns.tolist()}")
        return
    
    # Check that age column exists
    if 'age' not in alpha_df.columns:
        print(f"ERROR: 'age' column not found in {args.alpha_file}")
        print(f"Available columns: {alpha_df.columns.tolist()}")
        return
    
    # Generate output filenames from input
    base_name = os.path.basename(args.alpha_file).replace('_alpha_diversity_shannon.csv', '').replace('_alpha_diversity_adjusted.csv', '')
    
    # Plot alpha diversity
    alpha_title = f"{args.title_prefix} Alpha Diversity" if args.title_prefix else "Alpha Diversity"
    alpha_out = os.path.join(args.out_dir, f"{base_name}_alpha_scatter.png")
    plot_alpha_scatter(alpha_df, metric_name, alpha_title, alpha_out)
    
    # Plot beta diversity MDS
    beta_title = f"{args.title_prefix} Beta Diversity (MDS)" if args.title_prefix else "Beta Diversity (MDS)"
    beta_out = os.path.join(args.out_dir, f"{base_name}_beta_mds.png")
    plot_beta_mds(beta_df, alpha_df, beta_title, beta_out)

if __name__ == "__main__":
    main()
