#!/usr/bin/env python3
"""
Create MDS plots for mean and min beta diversity
- Both use same MDS positions (from distance matrix)
- Mean beta version: sized by mean beta
- Min beta version: sized by min beta
Point color = age in both
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
import glob
import os

def create_mds_plots_both_metrics(beta_file, alpha_file, output_prefix, title_base):
    """Create two MDS plots: one sized by mean, one sized by min"""
    
    beta_df = pd.read_csv(beta_file, index_col=0)
    alpha_df = pd.read_csv(alpha_file, index_col=0)
    
    # Calculate both metrics
    beta_copy = beta_df.copy()
    np.fill_diagonal(beta_copy.values, np.nan)
    
    mean_beta = beta_copy.mean(axis=1)
    min_beta = beta_copy.min(axis=1)
    
    # Create combined dataframe
    combined_df = pd.DataFrame({
        'animal': mean_beta.index,
        'mean_beta': mean_beta.values,
        'min_beta': min_beta.values,
        'age': alpha_df['age'].values
    })
    
    # Remove NaN
    n_before = len(combined_df)
    combined_df = combined_df.dropna()
    n_after = len(combined_df)
    
    if n_before != n_after:
        print(f"    Excluded {n_before - n_after} animals with NaN")
    
    if len(combined_df) < 3:
        print(f"  Skipping - not enough animals ({len(combined_df)})")
        return False
    
    # Filter beta matrix to only include animals without NaN
    valid_animals = combined_df['animal'].values
    beta_filtered = beta_df.loc[valid_animals, valid_animals]
    
    # MDS needs 0 on diagonal
    beta_mds = beta_filtered.copy()
    np.fill_diagonal(beta_mds.values, 0)
    
    # Run MDS (ONCE - same positions for both plots)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(beta_mds.values)
    
    # Create MEAN BETA plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Normalize mean beta for sizing
    mean_norm = (combined_df['mean_beta'].values - combined_df['mean_beta'].min()) / \
                (combined_df['mean_beta'].max() - combined_df['mean_beta'].min())
    sizes_mean = 100 + mean_norm * 400
    
    scatter = ax.scatter(coords[:, 0], coords[:, 1],
                        c=combined_df['age'].values,
                        s=sizes_mean,
                        cmap='viridis', alpha=0.7,
                        edgecolors='black', linewidth=0.5)
    
    # Add animal ID labels
    for i, animal_id in enumerate(combined_df['animal'].values):
        ax.annotate(animal_id, 
                   (coords[i, 0], coords[i, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8)
    
    plt.colorbar(scatter, ax=ax, label='Age (years)')
    
    # Add size legend
    for beta_val in [combined_df['mean_beta'].min(), 
                     combined_df['mean_beta'].median(),
                     combined_df['mean_beta'].max()]:
        beta_n = (beta_val - combined_df['mean_beta'].min()) / \
                 (combined_df['mean_beta'].max() - combined_df['mean_beta'].min())
        size = 100 + beta_n * 400
        ax.scatter([], [], s=size, c='gray', alpha=0.7, 
                  edgecolors='black', linewidth=0.5, label=f'{beta_val:.3f}')
    
    ax.legend(title='Mean Beta Diversity', loc='upper right', fontsize=9, framealpha=0.9)
    
    ax.set_xlabel('MDS Dimension 1', fontsize=12)
    ax.set_ylabel('MDS Dimension 2', fontsize=12)
    ax.set_title(f'{title_base} - MDS (sized by Mean Beta)', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_mds_mean_beta.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create MIN BETA plot (same positions, different sizes)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Normalize min beta for sizing
    min_norm = (combined_df['min_beta'].values - combined_df['min_beta'].min()) / \
               (combined_df['min_beta'].max() - combined_df['min_beta'].min())
    sizes_min = 100 + min_norm * 400
    
    scatter = ax.scatter(coords[:, 0], coords[:, 1],
                        c=combined_df['age'].values,
                        s=sizes_min,
                        cmap='viridis', alpha=0.7,
                        edgecolors='black', linewidth=0.5)
    
    # Add animal ID labels
    for i, animal_id in enumerate(combined_df['animal'].values):
        ax.annotate(animal_id, 
                   (coords[i, 0], coords[i, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8)
    
    plt.colorbar(scatter, ax=ax, label='Age (years)')
    
    # Add size legend
    for beta_val in [combined_df['min_beta'].min(), 
                     combined_df['min_beta'].median(),
                     combined_df['min_beta'].max()]:
        beta_n = (beta_val - combined_df['min_beta'].min()) / \
                 (combined_df['min_beta'].max() - combined_df['min_beta'].min())
        size = 100 + beta_n * 400
        ax.scatter([], [], s=size, c='gray', alpha=0.7, 
                  edgecolors='black', linewidth=0.5, label=f'{beta_val:.3f}')
    
    ax.legend(title='Min Beta Diversity', loc='upper right', fontsize=9, framealpha=0.9)
    
    ax.set_xlabel('MDS Dimension 1', fontsize=12)
    ax.set_ylabel('MDS Dimension 2', fontsize=12)
    ax.set_title(f'{title_base} - MDS (sized by Min Beta)', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_mds_min_beta.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return True

print("Creating MDS plots (mean and min beta versions)...\n")

# Process all beta diversity files
for beta_file in glob.glob('diversity/*_beta_diversity.csv'):
    base = os.path.basename(beta_file)
    
    # Get corresponding alpha file
    alpha_adj_file = beta_file.replace('_beta_diversity.csv', '_alpha_diversity_adjusted.csv')
    
    if not os.path.exists(alpha_adj_file):
        continue
    
    # Determine naming
    if '_celltype_' in base:
        region = base.replace('_celltype_beta_diversity.csv', '')
        prefix = f'diversity_figs/{region}_celltype'
        title_base = f'{region} Cell Types'
    else:  # subtype
        parts = base.replace('_subtype_beta_diversity.csv', '')
        region = parts.split('_')[0]
        celltype = '_'.join(parts.split('_')[1:]).replace('_', ' ')
        prefix = f'diversity_figs/{parts}'
        title_base = f'{region} {celltype}'
    
    print(f"Processing: {base}")
    
    success = create_mds_plots_both_metrics(beta_file, alpha_adj_file, prefix, title_base)
    if success:
        print(f"  ✓ Created both MDS plots")
    print()

print("\nDone! All MDS plots saved to diversity_figs/")
