#!/usr/bin/env python3
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def plot_stacked_bar(props_df, ages, title, out_file):
    """Plot stacked bar chart of cell type/subtype proportions"""
    fig, ax = plt.subplots(figsize=(16, 7))
    
    colors = plt.get_cmap("tab20").colors
    
    # Sort by age
    sort_idx = np.argsort(ages)
    props_sorted = props_df.iloc[sort_idx]
    ages_sorted = ages[sort_idx]
    
    bottom = np.zeros(len(props_sorted))
    for i, col in enumerate(props_sorted.columns):
        ax.bar(range(len(props_sorted)), props_sorted[col], bottom=bottom, 
               color=colors[i % len(colors)], label=col)
        bottom += props_sorted[col].values
    
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_xlabel("Individual", fontsize=12)
    ax.set_xticks(range(len(props_sorted)))
    
    # Create labels with animal ID and age
    labels = [f"{animal_id}, {age:.1f}" for animal_id, age in zip(props_sorted.index, ages_sorted)]
    ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=7)
    
    # Calculate number of columns for legend
    n_items = len(props_sorted.columns)
    ncol = min(5, max(3, n_items // 20))
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, ncol=ncol,
              frameon=False, columnspacing=0.5, handlelength=1, handletextpad=0.5)
    
    plt.title(title, fontsize=13)
    plt.subplots_adjust(bottom=0.15, right=0.85)
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Plot stacked bar composition")
    parser.add_argument("--region", type=str, required=True, help="Region name (e.g., HIP)")
    parser.add_argument("--level", type=str, choices=['celltype', 'subtype'], required=True)
    parser.add_argument("--celltype", type=str, default=None,
                        help="If level=subtype, which cell type")
    parser.add_argument("--celltype_col", type=str, default="cell_class_annotation")
    parser.add_argument("--subtype_col", type=str, default="subtype")
    parser.add_argument("--min_age", type=float, default=1.0)
    parser.add_argument("--h5ad_dir", type=str,
                        default="/scratch/nsnyderm/u01/intermediate_files/regions_h5ad_update")
    parser.add_argument("--out_dir", type=str,
                        default="/scratch/easmit31/variability/composition_figs")
    args = parser.parse_args()
    
    if args.level == 'subtype' and args.celltype is None:
        parser.error("--celltype is required when --level=subtype")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load data
    h5ad_file = os.path.join(args.h5ad_dir, f"{args.region}.h5ad")
    print(f"Loading {h5ad_file}...")
    adata = sc.read_h5ad(h5ad_file, backed='r')
    obs = adata.obs.copy()
    
    # Filter by age
    animal_ages = obs[['animal_id', 'age']].drop_duplicates()
    valid_animals = animal_ages[animal_ages['age'] >= args.min_age]['animal_id'].values
    obs = obs[obs['animal_id'].isin(valid_animals)]
    
    print(f"Plotting composition for {len(valid_animals)} individuals (age >= {args.min_age})")
    
    if args.level == 'celltype':
        counts = obs.groupby(['animal_id', args.celltype_col]).size().unstack(fill_value=0)
        prefix = f"{args.region}_celltype"
        title = f"{args.region} Cell Type Composition"
        
    else:  # subtype
        # Filter to only cells of the specified cell type
        obs_filtered = obs[obs[args.celltype_col] == args.celltype].copy()
        if len(obs_filtered) == 0:
            print(f"ERROR: No cells found for '{args.celltype}'")
            return
        
        # Get subtypes that belong ONLY to this cell type
        subtypes_in_celltype = obs_filtered[args.subtype_col].unique()
        print(f"Found {len(subtypes_in_celltype)} subtypes for {args.celltype}")
        
        counts = obs_filtered.groupby(['animal_id', args.subtype_col]).size().unstack(fill_value=0)
        
        # Keep only columns (subtypes) that belong to this cell type
        counts = counts[[col for col in counts.columns if col in subtypes_in_celltype]]
        
        celltype_clean = args.celltype.replace(' ', '_').replace('/', '_')
        prefix = f"{args.region}_{celltype_clean}_subtype"
        title = f"{args.region} {args.celltype} Subtype Composition"
    
    # Ensure only valid animals
    counts = counts.loc[counts.index.isin(valid_animals)]
    props = counts.div(counts.sum(axis=1), axis=0)
    
    # Get ages
    metadata = obs[['animal_id', 'age']].drop_duplicates().set_index('animal_id')
    ages = metadata.loc[props.index, 'age'].values
    
    # Plot
    out_file = os.path.join(args.out_dir, f"{prefix}_composition_bars.png")
    plot_stacked_bar(props, ages, title, out_file)

if __name__ == "__main__":
    main()
