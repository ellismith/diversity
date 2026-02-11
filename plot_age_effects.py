#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import argparse
import os

def plot_regression_scatter(alpha_file, title, out_file, stats_dict=None):
    """Scatter plot with regression line"""
    df = pd.read_csv(alpha_file, index_col=0)
    
    # Detect which metric
    if 'shannon_diversity' in df.columns:
        metric_col = 'shannon_diversity'
        ylabel = 'Shannon Diversity'
    elif 'adjusted_entropy' in df.columns:
        metric_col = 'adjusted_entropy'
        ylabel = 'Adjusted Entropy'
    else:
        print(f"ERROR: No recognized diversity metric in {alpha_file}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter points
    scatter = ax.scatter(df['age'], df[metric_col], 
                        c=df['age'], cmap='viridis', s=100, alpha=0.7, 
                        edgecolors='black', linewidth=0.5)
    
    # Regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['age'], df[metric_col])
    x_range = np.array([df['age'].min(), df['age'].max()])
    y_pred = slope * x_range + intercept
    ax.plot(x_range, y_pred, 'r--', linewidth=2, alpha=0.8, label='Linear fit')
    
    # Add stats to plot
    stats_text = f"r = {r_value:.3f}\n"
    stats_text += f"r² = {r_value**2:.3f}\n"
    stats_text += f"p = {p_value:.4f}\n"
    stats_text += f"slope = {slope:.4f} ± {std_err:.4f}"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Age (years)', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, pad=15)
    ax.grid(alpha=0.3, ls='--')
    ax.legend(fontsize=10)
    
    plt.colorbar(scatter, ax=ax, label='Age (years)')
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file}")

def plot_beta_regression(beta_file, alpha_file, title, out_file):
    """Scatter plot of mean Bray-Curtis vs age with regression"""
    beta_df = pd.read_csv(beta_file, index_col=0)
    alpha_df = pd.read_csv(alpha_file, index_col=0)
    
    # Calculate mean distance
    mean_distances = beta_df.mean(axis=1)
    data = pd.DataFrame({
        'age': alpha_df['age'],
        'mean_bray_curtis': mean_distances
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter points
    scatter = ax.scatter(data['age'], data['mean_bray_curtis'],
                        c=data['age'], cmap='viridis', s=100, alpha=0.7,
                        edgecolors='black', linewidth=0.5)
    
    # Regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(data['age'], data['mean_bray_curtis'])
    x_range = np.array([data['age'].min(), data['age'].max()])
    y_pred = slope * x_range + intercept
    ax.plot(x_range, y_pred, 'r--', linewidth=2, alpha=0.8, label='Linear fit')
    
    # Add stats
    stats_text = f"r = {r_value:.3f}\n"
    stats_text += f"r² = {r_value**2:.3f}\n"
    stats_text += f"p = {p_value:.4f}\n"
    stats_text += f"slope = {slope:.4f} ± {std_err:.4f}"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Age (years)', fontsize=13)
    ax.set_ylabel('Mean Bray-Curtis Distance', fontsize=13)
    ax.set_title(title, fontsize=14, pad=15)
    ax.grid(alpha=0.3, ls='--')
    ax.legend(fontsize=10)
    
    plt.colorbar(scatter, ax=ax, label='Age (years)')
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file}")

def plot_effect_size_summary(stats_file, out_file):
    """Forest plot of effect sizes (correlations) across cell types"""
    df = pd.read_csv(stats_file)
    
    # Separate alpha and beta
    alpha_df = df[df['metric'] == 'alpha_diversity'].copy()
    beta_df = df[df['metric'] == 'beta_diversity'].copy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Alpha diversity
    alpha_df = alpha_df.sort_values('r')
    y_pos = np.arange(len(alpha_df))
    colors_alpha = ['red' if p < 0.05 else 'gray' for p in alpha_df['p_value']]
    
    ax1.barh(y_pos, alpha_df['r'], color=colors_alpha, alpha=0.7)
    ax1.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(alpha_df['cell_type'], fontsize=9)
    ax1.set_xlabel('Correlation (r)', fontsize=12)
    ax1.set_title('Alpha Diversity ~ Age', fontsize=13)
    ax1.grid(alpha=0.3, axis='x')
    
    # Add p-values
    for i, (r, p) in enumerate(zip(alpha_df['r'], alpha_df['p_value'])):
        label = f"p={p:.3f}" if p >= 0.001 else f"p<0.001"
        ax1.text(r + 0.02 if r > 0 else r - 0.02, i, label, 
                va='center', ha='left' if r > 0 else 'right', fontsize=7)
    
    # Beta diversity
    beta_df = beta_df.sort_values('r')
    y_pos = np.arange(len(beta_df))
    colors_beta = ['red' if p < 0.05 else 'gray' for p in beta_df['p_value']]
    
    ax2.barh(y_pos, beta_df['r'], color=colors_beta, alpha=0.7)
    ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(beta_df['cell_type'], fontsize=9)
    ax2.set_xlabel('Correlation (r)', fontsize=12)
    ax2.set_title('Beta Diversity ~ Age', fontsize=13)
    ax2.grid(alpha=0.3, axis='x')
    
    # Add p-values
    for i, (r, p) in enumerate(zip(beta_df['r'], beta_df['p_value'])):
        label = f"p={p:.3f}" if p >= 0.001 else f"p<0.001"
        ax2.text(r + 0.02 if r > 0 else r - 0.02, i, label,
                va='center', ha='left' if r > 0 else 'right', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Plot age effects on diversity")
    parser.add_argument("--stats_file", type=str, required=True,
                        help="CSV file with age effect statistics")
    parser.add_argument("--diversity_dir", type=str, default="/scratch/easmit31/variability/diversity")
    parser.add_argument("--out_dir", type=str, default="/scratch/easmit31/variability/diversity_figs")
    parser.add_argument("--plot_individual", action='store_true',
                        help="Plot individual regression plots for each cell type")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load stats
    stats_df = pd.read_csv(args.stats_file)
    region = os.path.basename(args.stats_file).split('_')[0]
    level = os.path.basename(args.stats_file).split('_')[1]
    
    # Detect metric type from filename
    if 'adjusted' in args.stats_file:
        metric = 'adjusted'
    elif 'shannon' in args.stats_file:
        metric = 'shannon'
    else:
        metric = 'shannon'  # default
    
    # Create summary plot
    summary_out = os.path.join(args.out_dir, f"{region}_{level}_age_effects_summary_{metric}.png")
    plot_effect_size_summary(args.stats_file, summary_out)
    
    # Optionally plot individual regressions
    if args.plot_individual:
        for _, row in stats_df.iterrows():
            celltype = row['cell_type']
            metric_type = row['metric']
            
            if metric_type == 'alpha_diversity':
                celltype_clean = celltype.replace(' ', '_').replace('/', '_')
                if level == 'celltype':
                    prefix = f"{region}_celltype"
                else:
                    prefix = f"{region}_{celltype_clean}_subtype"
                
                alpha_file = os.path.join(args.diversity_dir, f"{prefix}_alpha_diversity_{metric}.csv")
                if not os.path.exists(alpha_file):
                    print(f"Skipping {celltype}: {alpha_file} not found")
                    continue
                
                title = f"{region} {celltype} - Alpha Diversity vs Age"
                out_file = os.path.join(args.out_dir, f"{prefix}_alpha_age_regression_{metric}.png")
                plot_regression_scatter(alpha_file, title, out_file)
            
            elif metric_type == 'beta_diversity':
                celltype_clean = celltype.replace(' ', '_').replace('/', '_')
                if level == 'celltype':
                    prefix = f"{region}_celltype"
                else:
                    prefix = f"{region}_{celltype_clean}_subtype"
                
                beta_file = os.path.join(args.diversity_dir, f"{prefix}_beta_diversity.csv")
                alpha_file = os.path.join(args.diversity_dir, f"{prefix}_alpha_diversity_{metric}.csv")
                if not os.path.exists(beta_file) or not os.path.exists(alpha_file):
                    print(f"Skipping {celltype}: files not found")
                    continue
                
                title = f"{region} {celltype} - Beta Diversity vs Age"
                out_file = os.path.join(args.out_dir, f"{prefix}_beta_age_regression_{metric}.png")
                plot_beta_regression(beta_file, alpha_file, title, out_file)

if __name__ == "__main__":
    main()
