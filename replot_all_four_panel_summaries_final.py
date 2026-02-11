#!/usr/bin/env python3
"""
Regenerate ALL four-panel forest plots with correct filtering
- Exclude animals without cells (check h5ad)
- Include animals with one subtype (diversity = 0 or -1)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scanpy as sc
import glob
import os

def get_animals_with_cells(region, celltype):
    """Get animals that have cells for this region/celltype"""
    h5ad_file = f'/scratch/nsnyderm/u01/intermediate_files/regions_h5ad_update/{region}.h5ad'
    adata = sc.read_h5ad(h5ad_file, backed='r')
    obs = adata.obs.copy()
    
    celltype_formatted = celltype.replace('_', ' ')
    obs_filtered = obs[obs['cell_class_annotation'] == celltype_formatted]
    animals = set(obs_filtered['animal_id'].unique())
    
    return animals

def compute_age_effects(alpha_file, animals_with_cells, metric_col):
    """Compute age effects for animals with cells"""
    df = pd.read_csv(alpha_file, index_col=0)
    
    # Keep only animals with cells
    df = df[df.index.isin(animals_with_cells)]
    
    # For Shannon: NaN means one subtype, replace with 0
    if metric_col == 'shannon_diversity':
        df[metric_col] = df[metric_col].fillna(0.0)
    
    if len(df) < 2:
        return None
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['age'], df[metric_col])
    
    return {
        'r': r_value,
        'p_value': p_value,
        'slope': slope,
        'std_err': std_err
    }

def compute_beta_age_effects(beta_file, alpha_file, animals_with_cells, metric='mean'):
    """Compute beta diversity age effects"""
    beta_df = pd.read_csv(beta_file, index_col=0)
    alpha_df = pd.read_csv(alpha_file, index_col=0)
    
    # Keep only animals with cells
    beta_df = beta_df.loc[beta_df.index.isin(animals_with_cells), 
                          beta_df.columns.isin(animals_with_cells)]
    alpha_df = alpha_df[alpha_df.index.isin(animals_with_cells)]
    
    beta_copy = beta_df.copy()
    np.fill_diagonal(beta_copy.values, np.nan)
    
    if metric == 'mean':
        beta_values = beta_copy.mean(axis=1)
    elif metric == 'min':
        beta_values = beta_copy.min(axis=1)
    
    combined_df = pd.DataFrame({
        'beta': beta_values.values,
        'age': alpha_df['age'].values
    })
    
    combined_df = combined_df.dropna()
    
    if len(combined_df) < 2:
        return None
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(combined_df['age'], combined_df['beta'])
    
    return {
        'r': r_value,
        'p_value': p_value,
        'slope': slope,
        'std_err': std_err
    }

def plot_four_panel_summary(region, level):
    """Create four-panel forest plot"""
    
    print(f"\n  Creating four-panel for {region} {level}")
    
    results = {
        'adjusted': [],
        'shannon': [],
        'mean_beta': [],
        'min_beta': []
    }
    
    if level == 'celltype':
        # Single region celltype
        alpha_adj_file = f'diversity/{region}_celltype_alpha_diversity_adjusted.csv'
        alpha_shan_file = f'diversity/{region}_celltype_alpha_diversity_shannon.csv'
        beta_file = f'diversity/{region}_celltype_beta_diversity.csv'
        
        if not all([os.path.exists(f) for f in [alpha_adj_file, alpha_shan_file, beta_file]]):
            print("    Missing files")
            return
        
        # All animals have data for celltype level
        alpha_df = pd.read_csv(alpha_adj_file, index_col=0)
        animals_with_cells = set(alpha_df.index)
        
        # Compute stats
        res = compute_age_effects(alpha_adj_file, animals_with_cells, 'adjusted_entropy')
        if res:
            results['adjusted'].append({'cell_type': region, 'r': res['r'], 'p_value': res['p_value']})
        
        res = compute_age_effects(alpha_shan_file, animals_with_cells, 'shannon_diversity')
        if res:
            results['shannon'].append({'cell_type': region, 'r': res['r'], 'p_value': res['p_value']})
        
        res = compute_beta_age_effects(beta_file, alpha_adj_file, animals_with_cells, 'mean')
        if res:
            results['mean_beta'].append({'cell_type': region, 'r': res['r'], 'p_value': res['p_value']})
        
        res = compute_beta_age_effects(beta_file, alpha_adj_file, animals_with_cells, 'min')
        if res:
            results['min_beta'].append({'cell_type': region, 'r': res['r'], 'p_value': res['p_value']})
    
    else:  # subtype
        subtype_files = glob.glob(f'diversity/{region}_*_subtype_alpha_diversity_adjusted.csv')
        
        for alpha_adj_file in subtype_files:
            base = os.path.basename(alpha_adj_file)
            parts = base.replace('_subtype_alpha_diversity_adjusted.csv', '').split('_')
            celltype = '_'.join(parts[1:])
            celltype_display = celltype.replace('_', ' ')
            
            alpha_shan_file = alpha_adj_file.replace('_adjusted.csv', '_shannon.csv')
            beta_file = alpha_adj_file.replace('_alpha_diversity_adjusted.csv', '_beta_diversity.csv')
            
            if not all([os.path.exists(f) for f in [alpha_shan_file, beta_file]]):
                continue
            
            # Get animals with cells
            try:
                animals_with_cells = get_animals_with_cells(region, celltype)
            except:
                continue
            
            # Compute stats
            res = compute_age_effects(alpha_adj_file, animals_with_cells, 'adjusted_entropy')
            if res:
                results['adjusted'].append({'cell_type': celltype_display, 'r': res['r'], 'p_value': res['p_value']})
            
            res = compute_age_effects(alpha_shan_file, animals_with_cells, 'shannon_diversity')
            if res:
                results['shannon'].append({'cell_type': celltype_display, 'r': res['r'], 'p_value': res['p_value']})
            
            res = compute_beta_age_effects(beta_file, alpha_adj_file, animals_with_cells, 'mean')
            if res:
                results['mean_beta'].append({'cell_type': celltype_display, 'r': res['r'], 'p_value': res['p_value']})
            
            res = compute_beta_age_effects(beta_file, alpha_adj_file, animals_with_cells, 'min')
            if res:
                results['min_beta'].append({'cell_type': celltype_display, 'r': res['r'], 'p_value': res['p_value']})
    
    # Convert to dataframes
    df_adj = pd.DataFrame(results['adjusted'])
    df_shan = pd.DataFrame(results['shannon'])
    df_mean = pd.DataFrame(results['mean_beta'])
    df_min = pd.DataFrame(results['min_beta'])
    
    if len(df_adj) == 0:
        print("    No data")
        return
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    def add_panel(ax, data, title):
        if len(data) == 0:
            return
        
        data = data.sort_values('r')
        y_pos = np.arange(len(data))
        colors = ['red' if p < 0.05 else 'gray' for p in data['p_value']]
        
        ax.barh(y_pos, data['r'], color=colors, alpha=0.7)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(data['cell_type'], fontsize=9)
        ax.set_xlabel('Correlation (r)', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')
        
        xlim = ax.get_xlim()
        x_range = xlim[1] - xlim[0]
        
        for i, (r, p) in enumerate(zip(data['r'], data['p_value'])):
            label = f"p={p:.3f}" if p >= 0.001 else f"p<0.001"
            
            if r > 0:
                x_pos = r + 0.05 * x_range
                ha = 'left'
            else:
                x_pos = r - 0.05 * x_range
                ha = 'right'
            
            ax.text(x_pos, i, label, va='center', ha=ha, fontsize=7,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    add_panel(ax1, df_adj, 'Adjusted Entropy ~ Age')
    add_panel(ax2, df_shan, 'Shannon Diversity ~ Age')
    add_panel(ax3, df_mean, 'Beta Diversity (Mean) ~ Age')
    add_panel(ax4, df_min, 'Beta Diversity (Min) ~ Age')
    
    plt.tight_layout()
    output_file = f'diversity_figs/{region}_{level}_age_effects_four_panel.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_file}")

print("Regenerating ALL four-panel plots...\n")

# Celltype level
celltype_files = glob.glob('diversity/*_celltype_age_effects_adjusted.csv')
print("Processing celltype level...")
for adj_file in celltype_files:
    base = os.path.basename(adj_file)
    region = base.replace('_celltype_age_effects_adjusted.csv', '')
    plot_four_panel_summary(region, 'celltype')

# Subtype level
subtype_files = glob.glob('diversity/*_subtype_age_effects_adjusted.csv')
print("\nProcessing subtype level...")
for adj_file in subtype_files:
    base = os.path.basename(adj_file)
    region = base.replace('_subtype_age_effects_adjusted.csv', '')
    plot_four_panel_summary(region, 'subtype')

print("\nDone!")
