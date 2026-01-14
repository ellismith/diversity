#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy import stats
import argparse
import os
import glob

def test_age_effect(alpha_file):
    """Test linear relationship between age and Shannon diversity"""
    df = pd.read_csv(alpha_file, index_col=0)
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['age'], df['shannon_diversity'])
    
    # Pearson correlation
    corr, corr_p = stats.pearsonr(df['age'], df['shannon_diversity'])
    
    return {
        'n': len(df),
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'r': corr,
        'p_value': p_value,
        'std_err': std_err
    }

def test_beta_age_effect(beta_file, alpha_file):
    """Test relationship between age and mean Bray-Curtis distance"""
    beta_df = pd.read_csv(beta_file, index_col=0)
    alpha_df = pd.read_csv(alpha_file, index_col=0)
    
    # Calculate mean distance for each individual
    mean_distances = beta_df.mean(axis=1)
    
    # Merge with ages
    data = pd.DataFrame({
        'age': alpha_df['age'],
        'mean_bray_curtis': mean_distances
    })
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(data['age'], data['mean_bray_curtis'])
    
    # Pearson correlation
    corr, corr_p = stats.pearsonr(data['age'], data['mean_bray_curtis'])
    
    return {
        'n': len(data),
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'r': corr,
        'p_value': p_value,
        'std_err': std_err
    }

def main():
    parser = argparse.ArgumentParser(description="Test age effects on diversity metrics")
    parser.add_argument("--region", type=str, required=True, help="Region name (e.g., HIP, CN)")
    parser.add_argument("--level", type=str, choices=['celltype', 'subtype'], default='subtype')
    parser.add_argument("--diversity_dir", type=str, default="/scratch/easmit31/variability/diversity")
    parser.add_argument("--out_dir", type=str, default="/scratch/easmit31/variability/diversity")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Find all relevant files
    if args.level == 'celltype':
        alpha_pattern = f"{args.region}_celltype_alpha_diversity.csv"
        beta_pattern = f"{args.region}_celltype_beta_diversity.csv"
        alpha_files = [os.path.join(args.diversity_dir, alpha_pattern)]
        beta_files = [os.path.join(args.diversity_dir, beta_pattern)]
    else:
        alpha_pattern = f"{args.region}_*_subtype_alpha_diversity.csv"
        beta_pattern = f"{args.region}_*_subtype_beta_diversity.csv"
        alpha_files = sorted(glob.glob(os.path.join(args.diversity_dir, alpha_pattern)))
        beta_files = sorted(glob.glob(os.path.join(args.diversity_dir, beta_pattern)))
    
    # Test alpha diversity
    alpha_results = []
    for alpha_file in alpha_files:
        if not os.path.exists(alpha_file):
            continue
        
        basename = os.path.basename(alpha_file)
        if args.level == 'celltype':
            celltype = 'All Cell Types'
        else:
            celltype = basename.replace(f"{args.region}_", "").replace("_subtype_alpha_diversity.csv", "")
            celltype = celltype.replace("_", " ")
        
        result = test_age_effect(alpha_file)
        result['cell_type'] = celltype
        result['metric'] = 'alpha_diversity'
        alpha_results.append(result)
        
        print(f"\n{celltype} - Alpha Diversity:")
        print(f"  n={result['n']}, r={result['r']:.3f}, r²={result['r_squared']:.3f}, p={result['p_value']:.4f}")
        print(f"  slope={result['slope']:.4f} ± {result['std_err']:.4f}")
    
    # Test beta diversity
    beta_results = []
    for beta_file, alpha_file in zip(beta_files, alpha_files):
        if not os.path.exists(beta_file) or not os.path.exists(alpha_file):
            continue
        
        basename = os.path.basename(beta_file)
        if args.level == 'celltype':
            celltype = 'All Cell Types'
        else:
            celltype = basename.replace(f"{args.region}_", "").replace("_subtype_beta_diversity.csv", "")
            celltype = celltype.replace("_", " ")
        
        result = test_beta_age_effect(beta_file, alpha_file)
        result['cell_type'] = celltype
        result['metric'] = 'beta_diversity'
        beta_results.append(result)
        
        print(f"\n{celltype} - Beta Diversity:")
        print(f"  n={result['n']}, r={result['r']:.3f}, r²={result['r_squared']:.3f}, p={result['p_value']:.4f}")
        print(f"  slope={result['slope']:.4f} ± {result['std_err']:.4f}")
    
    # Save results
    all_results = alpha_results + beta_results
    results_df = pd.DataFrame(all_results)
    
    out_file = os.path.join(args.out_dir, f"{args.region}_{args.level}_age_effects.csv")
    results_df.to_csv(out_file, index=False)
    print(f"\n\nSaved all results to: {out_file}")

if __name__ == "__main__":
    main()
