#!/usr/bin/env python3
"""
Replot alpha diversity - FINAL CORRECT VERSION
Only include animals that actually have cells for this region/celltype combo
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scanpy as sc
import os

cell_types = [
    ('IPP', 'oligodendrocyte_precursor_cells'),
    ('lCb', 'astrocytes'),
    ('lCb', 'microglia'),
    ('lCb', 'oligodendrocyte_precursor_cells'),
    ('MB', 'oligodendrocyte_precursor_cells'),
    ('mdTN', 'GABAergic_neurons'),
    ('NAc', 'glutamatergic_neurons'),
    ('NAc', 'microglia'),
    ('NAc', 'oligodendrocyte_precursor_cells')
]

def get_animals_with_cells(region, celltype):
    """Get list of animals that actually have cells for this region/celltype"""
    h5ad_file = f'/scratch/nsnyderm/u01/intermediate_files/regions_h5ad_update/{region}.h5ad'
    adata = sc.read_h5ad(h5ad_file, backed='r')
    obs = adata.obs.copy()
    
    # Convert celltype format
    celltype_formatted = celltype.replace('_', ' ')
    
    # Filter to this cell type
    obs_filtered = obs[obs['cell_class_annotation'] == celltype_formatted]
    
    # Get animals that have at least 1 cell
    animals_with_cells = obs_filtered['animal_id'].unique()
    
    return set(animals_with_cells)

def plot_alpha_regression(alpha_file, animals_with_cells, output_file, title, metric_col, ylabel):
    alpha_df = pd.read_csv(alpha_file, index_col=0)
    
    # Keep only animals that have cells
    n_before = len(alpha_df)
    alpha_df = alpha_df[alpha_df.index.isin(animals_with_cells)]
    n_after = len(alpha_df)
    
    if n_before != n_after:
        print(f"    Excluded {n_before - n_after} animals without cells")
    
    # For Shannon: NaN means one subtype, replace with 0
    if metric_col == 'shannon_diversity':
        alpha_df[metric_col] = alpha_df[metric_col].fillna(0.0)
    
    print(f"    Using {len(alpha_df)} animals")
    
    if len(alpha_df) < 2:
        return False
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(alpha_df['age'], alpha_df[metric_col])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(alpha_df['age'], alpha_df[metric_col],
                        c=alpha_df['age'], cmap='viridis', s=100, alpha=0.7,
                        edgecolors='black', linewidth=0.5)
    
    x_range = np.array([alpha_df['age'].min(), alpha_df['age'].max()])
    y_pred = slope * x_range + intercept
    ax.plot(x_range, y_pred, 'r--', linewidth=2, alpha=0.8)
    
    stats_text = f"r = {r_value:.3f}\nr² = {r_value**2:.3f}\np = {p_value:.4f}\nslope = {slope:.4f} ± {std_err:.4f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Age (years)', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, pad=15)
    ax.grid(alpha=0.3, ls='--')
    plt.colorbar(scatter, ax=ax, label='Age (years)')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    return True

print("Replotting alpha diversity (FINAL - checking actual cells)...\n")

for region, celltype in cell_types:
    print(f"{region} - {celltype}")
    
    # Get animals that actually have cells
    try:
        animals_with_cells = get_animals_with_cells(region, celltype)
        print(f"  Found {len(animals_with_cells)} animals with cells")
    except Exception as e:
        print(f"  Error getting animals: {e}")
        continue
    
    # Adjusted
    alpha_adj_file = f'diversity/{region}_{celltype}_subtype_alpha_diversity_adjusted.csv'
    if os.path.exists(alpha_adj_file):
        output_file = f'diversity_figs/{region}_{celltype}_alpha_regression_adjusted.png'
        celltype_display = celltype.replace('_', ' ')
        title = f'{region} {celltype_display} - Adjusted Entropy vs Age'
        plot_alpha_regression(alpha_adj_file, animals_with_cells, output_file, title, 'adjusted_entropy', 'Adjusted Entropy')
    
    # Shannon
    alpha_shan_file = f'diversity/{region}_{celltype}_subtype_alpha_diversity_shannon.csv'
    if os.path.exists(alpha_shan_file):
        output_file = f'diversity_figs/{region}_{celltype}_alpha_regression_shannon.png'
        celltype_display = celltype.replace('_', ' ')
        title = f'{region} {celltype_display} - Shannon Diversity vs Age'
        plot_alpha_regression(alpha_shan_file, animals_with_cells, output_file, title, 'shannon_diversity', 'Shannon Diversity')
    
    print()

print("Done!")
