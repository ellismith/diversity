#!/usr/bin/env python3
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def plot_stacked_bar(df, colors, title, out_file, ages):
    fig, ax = plt.subplots(figsize=(16,7))
    bottom = np.zeros(len(df))
    for i, col in enumerate(df.columns):
        ax.bar(df.index, df[col], bottom=bottom, color=colors[i % len(colors)], label=col)
        bottom += df[col].values
    
    ax.set_ylabel("Proportion", fontsize=11)
    ax.set_xlabel("Individual", fontsize=11)
    ax.set_xticks(range(len(df)))
    # Create single-line labels with animal ID and age separated by comma
    labels = [f"{animal_id}, {age}" for animal_id, age in zip(df.index, ages)]
    ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=7)
    
    # Calculate number of columns based on number of legend items
    n_items = len(df.columns)
    ncol = min(5, max(3, n_items // 20))  # 3-5 columns depending on number of items
    
    ax.legend(bbox_to_anchor=(1.02,1), loc='upper left', fontsize=6, ncol=ncol, 
              frameon=False, columnspacing=0.5, handlelength=1, handletextpad=0.5)
    plt.title(title, fontsize=12)
    plt.subplots_adjust(bottom=0.15, right=0.85)
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plot cell type composition for a region")
    parser.add_argument("--region_file", type=str, required=True, help="H5AD file for region (e.g., dlPFC.h5ad)")
    parser.add_argument("--celltype_col", type=str, default="cell_class_annotation")
    parser.add_argument("--subtype_col", type=str, default="subtype")
    parser.add_argument("--out_dir", type=str, default="/scratch/easmit31/variability/composition_figs")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    adata = sc.read_h5ad(args.region_file, backed='r')
    
    # --- Region-level proportions ---
    region_props = (
        adata.obs.groupby(["animal_id", args.celltype_col]).size().unstack(fill_value=0)
    )
    region_props = region_props.div(region_props.sum(axis=1), axis=0)
    
    # add ages for sorting
    ages = adata.obs[["animal_id","age"]].drop_duplicates().set_index("animal_id")
    region_props = region_props.merge(ages, left_index=True, right_index=True)
    region_props = region_props.sort_values("age")
    ages_sorted = region_props["age"].values
    region_props = region_props.drop(columns="age")
    
    colors = plt.get_cmap("tab20").colors
    out_file_region = os.path.join(args.out_dir, f"composition_region_{os.path.basename(args.region_file).replace('.h5ad','')}.png")
    plot_stacked_bar(region_props, colors, "Region-level cell type composition", out_file_region, ages_sorted)
    print(f"Saved region-level plot to {out_file_region}")
    
    # --- Within-cell-type proportions (using subtype) ---
    for ct in adata.obs[args.celltype_col].unique():
        # Filter to only this cell type
        obs_sub = adata.obs[adata.obs[args.celltype_col]==ct].copy()
        
        # Get subtypes that belong to this cell type only
        subtypes_in_ct = obs_sub[args.subtype_col].unique()
        
        subtype_props = (
            obs_sub.groupby(["animal_id", args.subtype_col]).size().unstack(fill_value=0)
        )
        subtype_props = subtype_props.div(subtype_props.sum(axis=1), axis=0)
        
        # Only keep columns (subtypes) that are in this cell type
        subtype_props = subtype_props[[col for col in subtype_props.columns if col in subtypes_in_ct]]
        
        ages_sub = obs_sub[["animal_id","age"]].drop_duplicates().set_index("animal_id")
        subtype_props = subtype_props.merge(ages_sub, left_index=True, right_index=True)
        subtype_props = subtype_props.sort_values("age")
        ages_sorted_sub = subtype_props["age"].values
        subtype_props = subtype_props.drop(columns="age")
        
        out_file_subtype = os.path.join(args.out_dir, f"composition_{ct}_{os.path.basename(args.region_file).replace('.h5ad','')}.png")
        plot_stacked_bar(subtype_props, colors, f"{ct} subtype composition", out_file_subtype, ages_sorted_sub)
        print(f"Saved {ct} plot to {out_file_subtype}")

if __name__ == "__main__":
    main()
