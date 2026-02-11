#!/usr/bin/env python3
"""
Compute age effect statistics for minimum beta diversity
Creates age_effects CSV files like we have for adjusted and shannon
"""

import pandas as pd
import numpy as np
from scipy import stats
import glob
import os

def compute_age_effects(min_beta_file):
    """Compute correlation between min beta diversity and age"""
    
    df = pd.read_csv(min_beta_file, index_col=0)
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['age'], df['min_beta_diversity'])
    
    return {
        'r': r_value,
        'p_value': p_value,
        'slope': slope,
        'std_err': std_err
    }

print("Computing age effects for minimum beta diversity...\n")

# Process celltype level
print("Processing celltype level...")
celltype_files = glob.glob('diversity/*_celltype_min_beta_diversity.csv')

for region_file in celltype_files:
    base = os.path.basename(region_file)
    region = base.replace('_celltype_min_beta_diversity.csv', '')
    
    result = compute_age_effects(region_file)
    
    # Create age effects dataframe
    age_effects_df = pd.DataFrame([{
        'cell_type': region,
        'metric': 'beta_diversity',
        'r': result['r'],
        'p_value': result['p_value'],
        'slope': result['slope'],
        'std_err': result['std_err']
    }])
    
    # Save
    output_file = f'diversity/{region}_celltype_age_effects_min_beta.csv'
    age_effects_df.to_csv(output_file, index=False)
    print(f"  {region}: r={result['r']:.3f}, p={result['p_value']:.4f}")

# Process subtype level
print("\nProcessing subtype level...")
subtype_files = glob.glob('diversity/*_subtype_min_beta_diversity.csv')

# Group by region
regions = {}
for subtype_file in subtype_files:
    base = os.path.basename(subtype_file)
    parts = base.replace('_subtype_min_beta_diversity.csv', '').split('_')
    region = parts[0]
    celltype = '_'.join(parts[1:])
    
    if region not in regions:
        regions[region] = []
    
    result = compute_age_effects(subtype_file)
    regions[region].append({
        'cell_type': celltype.replace('_', ' '),
        'metric': 'beta_diversity',
        'r': result['r'],
        'p_value': result['p_value'],
        'slope': result['slope'],
        'std_err': result['std_err']
    })

# Save subtype age effects per region
for region, results in regions.items():
    age_effects_df = pd.DataFrame(results)
    output_file = f'diversity/{region}_subtype_age_effects_min_beta.csv'
    age_effects_df.to_csv(output_file, index=False)
    print(f"  {region}: {len(results)} subtypes processed")

print("\nDone! All min beta age effects computed.")
