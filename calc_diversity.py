#!/usr/bin/env python3
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
import os
import argparse

def calculate_alpha_diversity(props_df):
    """Calculate Shannon diversity for each individual (row)"""
    shannon = props_df.apply(lambda row: entropy(row, base=np.e), axis=1)
    return pd.DataFrame({'shannon_diversity': shannon})

def calculate_beta_diversity(props_df):
    """Calculate pairwise Bray-Curtis dissimilarity between individuals"""
    distances = pdist(props_df.values, metric='braycurtis')
    dist_matrix = squareform(distances)
    return pd.DataFrame(dist_matrix, index=props_df.index, columns=props_df.index)

def main():
    parser = argparse.ArgumentParser(description="Calculate alpha and beta diversity")
    parser.add_argument("--region", type=str, required=True, help="Region name (e.g., HIP, dlPFC)")
    parser.add_argument("--level", type=str, choices=['celltype', 'subtype'], required=True,
                        help="Calculate diversity at cell type or subtype level")
    parser.add_argument("--celltype", type=str, default=None,
                        help="If level=subtype, which cell type to analyze (e.g., 'GABAergic neurons')")
    parser.add_argument("--celltype_col", type=str, default="cell_class_annotation")
    parser.add_argument("--subtype_col", type=str, default="subtype")
    parser.add_argument("--min_age", type=float, default=1.0,
                        help="Minimum age to include (default: 1.0)")
    parser.add_argument("--h5ad_dir", type=str, 
                        default="/scratch/nsnyderm/u01/intermediate_files/regions_h5ad_update")
    parser.add_argument("--out_dir", type=str, default="/scratch/easmit31/variability/diversity")
    args = parser.parse_args()
    
    if args.level == 'subtype' and args.celltype is None:
        parser.error("--celltype is required when --level=subtype")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load data
    h5ad_file = os.path.join(args.h5ad_dir, f"{args.region}.h5ad")
    print(f"Loading {h5ad_file}...")
    adata = sc.read_h5ad(h5ad_file, backed='r')
    obs = adata.obs.copy()
    
    # Get valid animal IDs based on age filter
    animal_ages = obs[['animal_id', 'age']].drop_duplicates()
    valid_animals = animal_ages[animal_ages['age'] >= args.min_age]['animal_id'].values
    
    n_before = len(animal_ages)
    n_after = len(valid_animals)
    print(f"Filtered to age >= {args.min_age}: {n_before} -> {n_after} individuals")
    
    # Filter obs to only valid animals
    obs = obs[obs['animal_id'].isin(valid_animals)]
    
    # Extract metadata (animal_id -> age) for valid animals only
    metadata = obs[['animal_id', 'age']].drop_duplicates().set_index('animal_id')
    
    if args.level == 'celltype':
        # Cell type diversity
        print(f"\nCalculating cell type diversity for {args.region}...")
        counts = obs.groupby(['animal_id', args.celltype_col]).size().unstack(fill_value=0)
        
        prefix = f"{args.region}_celltype"
        
    else:  # subtype
        # Filter to specific cell type
        print(f"\nCalculating subtype diversity for {args.celltype} in {args.region}...")
        obs_filtered = obs[obs[args.celltype_col] == args.celltype].copy()
        
        if len(obs_filtered) == 0:
            print(f"ERROR: No cells found for cell type '{args.celltype}'")
            return
        
        counts = obs_filtered.groupby(['animal_id', args.subtype_col]).size().unstack(fill_value=0)
        
        # Clean celltype name for filename
        celltype_clean = args.celltype.replace(' ', '_').replace('/', '_')
        prefix = f"{args.region}_{celltype_clean}_subtype"
    
    # Ensure only valid animals are in counts
    counts = counts.loc[counts.index.isin(valid_animals)]
    
    # Calculate proportions
    props = counts.div(counts.sum(axis=1), axis=0)
    
    print(f"Computing diversity for {len(props)} individuals")
    
    # Calculate alpha diversity
    alpha = calculate_alpha_diversity(props)
    # Add age to alpha diversity dataframe
    alpha = alpha.merge(metadata, left_index=True, right_index=True)
    alpha_file = os.path.join(args.out_dir, f"{prefix}_alpha_diversity.csv")
    alpha.to_csv(alpha_file)
    print(f"Saved: {alpha_file}")
    print(f"  Mean Shannon: {alpha['shannon_diversity'].mean():.3f}")
    print(f"  SD Shannon: {alpha['shannon_diversity'].std():.3f}")
    
    # Calculate beta diversity
    beta = calculate_beta_diversity(props)
    beta_file = os.path.join(args.out_dir, f"{prefix}_beta_diversity.csv")
    beta.to_csv(beta_file)
    print(f"Saved: {beta_file}")
    
    # Get upper triangle for mean (exclude diagonal)
    upper_tri = beta.values[np.triu_indices_from(beta.values, k=1)]
    print(f"  Mean Bray-Curtis: {upper_tri.mean():.3f}")
    print(f"  SD Bray-Curtis: {upper_tri.std():.3f}")
    
    # Save metadata
    metadata_file = os.path.join(args.out_dir, f"{prefix}_metadata.csv")
    metadata.to_csv(metadata_file)
    print(f"Saved: {metadata_file}")
    
    print(f"\nOutputs saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
