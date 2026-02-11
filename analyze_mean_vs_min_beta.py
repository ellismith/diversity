#!/usr/bin/env python3
"""
Analyze differences between mean and min beta diversity
for specific cell types where they show different patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import squareform
import os

# Cell types to analyze
cell_types = [
    ('dlPFC', 'microglia'),
    ('dlPFC', 'oligodendrocytes'),
    ('EC', 'oligodendrocytes'),
    ('HIP', 'GABAergic_neurons'),
    ('mdTN', 'microglia'),
    ('NAc', 'astrocytes'),
    ('NAc', 'glutamatergic_neurons')
]

output_dir = 'diversity_figs/mean_vs_min_analysis'
os.makedirs(output_dir, exist_ok=True)

for region, celltype in cell_types:
    print(f"\n{'='*60}")
    print(f"Analyzing: {region} - {celltype}")
    print(f"{'='*60}")
    
    # Load files
    beta_file = f'diversity/{region}_{celltype}_subtype_beta_diversity.csv'
    alpha_file = f'diversity/{region}_{celltype}_subtype_alpha_diversity_adjusted.csv'
    
    if not os.path.exists(beta_file):
        print(f"  File not found: {beta_file}")
        continue
    
    beta_df = pd.read_csv(beta_file, index_col=0)
    alpha_df = pd.read_csv(alpha_file, index_col=0)
    
    # Calculate mean and min beta
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
    
    # Remove any NaN
    combined_df = combined_df.dropna()
    
    print(f"\nN animals: {len(combined_df)}")
    
    # Stats
    mean_age_r, mean_age_p = stats.pearsonr(combined_df['age'], combined_df['mean_beta'])
    min_age_r, min_age_p = stats.pearsonr(combined_df['age'], combined_df['min_beta'])
    mean_min_r, mean_min_p = stats.pearsonr(combined_df['mean_beta'], combined_df['min_beta'])
    
    print(f"\nCorrelations:")
    print(f"  Mean beta ~ Age: r={mean_age_r:.3f}, p={mean_age_p:.4f}")
    print(f"  Min beta ~ Age:  r={min_age_r:.3f}, p={min_age_p:.4f}")
    print(f"  Mean ~ Min:      r={mean_min_r:.3f}, p={mean_min_p:.4f}")
    
    # Create 4-panel figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Distribution of all pairwise distances
    ax1 = fig.add_subplot(gs[0, :])
    
    # Get upper triangle of distances (exclude diagonal)
    upper_tri = beta_copy.values[np.triu_indices_from(beta_copy.values, k=1)]
    upper_tri = upper_tri[~np.isnan(upper_tri)]
    
    ax1.hist(upper_tri, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(upper_tri), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(upper_tri):.3f}')
    ax1.axvline(np.median(upper_tri), color='blue', linestyle='--', linewidth=2, label=f'Median = {np.median(upper_tri):.3f}')
    ax1.set_xlabel('Bray-Curtis Distance', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'{region} {celltype.replace("_", " ")} - Distribution of All Pairwise Distances', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Panel 2: Mean beta vs Min beta scatter
    ax2 = fig.add_subplot(gs[1, 0])
    
    scatter = ax2.scatter(combined_df['mean_beta'], combined_df['min_beta'],
                         c=combined_df['age'], cmap='viridis', s=100, alpha=0.7,
                         edgecolors='black', linewidth=0.5)
    
    # Add regression line
    slope, intercept = np.polyfit(combined_df['mean_beta'], combined_df['min_beta'], 1)
    x_line = np.array([combined_df['mean_beta'].min(), combined_df['mean_beta'].max()])
    y_line = slope * x_line + intercept
    ax2.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Mean Beta Diversity', fontsize=11)
    ax2.set_ylabel('Min Beta Diversity', fontsize=11)
    ax2.set_title(f'Mean vs Min Beta (r={mean_min_r:.3f})', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Age')
    
    # Panel 3: Mean beta vs Age
    ax3 = fig.add_subplot(gs[1, 1])
    
    scatter = ax3.scatter(combined_df['age'], combined_df['mean_beta'],
                         c=combined_df['age'], cmap='viridis', s=100, alpha=0.7,
                         edgecolors='black', linewidth=0.5)
    
    slope, intercept = np.polyfit(combined_df['age'], combined_df['mean_beta'], 1)
    x_line = np.array([combined_df['age'].min(), combined_df['age'].max()])
    y_line = slope * x_line + intercept
    ax3.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.8)
    
    ax3.set_xlabel('Age (years)', fontsize=11)
    ax3.set_ylabel('Mean Beta Diversity', fontsize=11)
    ax3.set_title(f'Mean Beta vs Age (r={mean_age_r:.3f}, p={mean_age_p:.4f})', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # Panel 4: Min beta vs Age
    ax4 = fig.add_subplot(gs[2, :])
    
    scatter = ax4.scatter(combined_df['age'], combined_df['min_beta'],
                         c=combined_df['age'], cmap='viridis', s=100, alpha=0.7,
                         edgecolors='black', linewidth=0.5)
    
    slope, intercept = np.polyfit(combined_df['age'], combined_df['min_beta'], 1)
    x_line = np.array([combined_df['age'].min(), combined_df['age'].max()])
    y_line = slope * x_line + intercept
    ax4.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.8)
    
    ax4.set_xlabel('Age (years)', fontsize=11)
    ax4.set_ylabel('Min Beta Diversity (Uniqueness)', fontsize=11)
    ax4.set_title(f'Min Beta vs Age (r={min_age_r:.3f}, p={min_age_p:.4f})', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Age')
    
    # Save
    plt.suptitle(f'{region} {celltype.replace("_", " ")} - Mean vs Min Beta Analysis', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    output_file = f'{output_dir}/{region}_{celltype}_mean_vs_min_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")

print(f"\n\nAll analyses saved to {output_dir}/")
