#!/usr/bin/env python3
"""
Create minimum beta diversity CSVs from existing beta diversity matrices
Minimum distance = uniqueness measure (distance to nearest neighbor)
FIXED: Handle cases where animals have no valid neighbors (isolated samples)
"""

import pandas as pd
import numpy as np
import glob
import os

print("Creating minimum beta diversity CSVs (fixed version)...\n")

# Get all beta diversity files (both celltype and subtype)
beta_files = glob.glob('diversity/*_beta_diversity.csv')

for beta_file in beta_files:
    print(f"Processing: {beta_file}")
    
    # Load beta diversity matrix
    beta_df = pd.read_csv(beta_file, index_col=0)
    
    # Create a copy and set diagonal to NaN (exclude self-comparison)
    beta_copy = beta_df.copy()
    np.fill_diagonal(beta_copy.values, np.nan)
    
    # Get minimum distance for each animal (nearest neighbor)
    min_distances = beta_copy.min(axis=1)
    
    # Get corresponding alpha file to get age information
    alpha_file = beta_file.replace('_beta_diversity.csv', '_alpha_diversity_adjusted.csv')
    
    if os.path.exists(alpha_file):
        alpha_df = pd.read_csv(alpha_file, index_col=0)
        
        # Create min beta dataframe
        min_beta_df = pd.DataFrame({
            'animal_id': min_distances.index,
            'min_beta_diversity': min_distances.values,
            'age': alpha_df['age'].values
        })
        min_beta_df.set_index('animal_id', inplace=True)
        
        # Check for NaN values (animals with no valid neighbors)
        n_nan = min_beta_df['min_beta_diversity'].isna().sum()
        if n_nan > 0:
            print(f"  WARNING: {n_nan} animals have no valid neighbors (all distances are NaN)")
            print(f"  These animals will be EXCLUDED from the output:")
            nan_animals = min_beta_df[min_beta_df['min_beta_diversity'].isna()].index.tolist()
            print(f"  {nan_animals}")
            
            # Remove animals with NaN min_beta_diversity
            min_beta_df = min_beta_df.dropna(subset=['min_beta_diversity'])
        
        # Save
        output_file = beta_file.replace('_beta_diversity.csv', '_min_beta_diversity.csv')
        min_beta_df.to_csv(output_file)
        print(f"  Saved: {output_file} ({len(min_beta_df)} animals)")
    else:
        print(f"  Warning: No matching alpha file found for {beta_file}")

print("\nDone! All minimum beta diversity files created (fixed).")
