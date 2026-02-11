# Cell Type Diversity Analysis

Analysis pipeline for calculating and visualizing alpha and beta diversity of cell types and subtypes across brain regions in aging primates.

## Overview

This pipeline computes diversity metrics at two levels:
1. **Cell type level**: Diversity of major cell types within each brain region
2. **Subtype level**: Diversity of subtypes within each cell type (e.g., GABAergic neuron subtypes)

## Key Metrics

### Alpha Diversity (within-individual variation)
- **Adjusted Entropy**: Normalized Shannon entropy, range [-1, 0] where 0 = maximum diversity, -1 = minimum diversity (single type)
- **Shannon Diversity**: Standard Shannon entropy, higher values = more diversity

### Beta Diversity (between-individual variation)
- **Mean Bray-Curtis Distance**: Average dissimilarity between an individual and all others
- **Min Bray-Curtis Distance (Uniqueness)**: Distance to nearest neighbor, measures how divergent an individual is from the most similar individual

## Important Data Handling Notes

### Missing Data vs. Low Diversity
The pipeline distinguishes between two cases:
1. **Missing data**: Animal has NO cells for a specific region/celltype combination → EXCLUDED from analysis
2. **Low diversity**: Animal has cells but only 1 subtype → INCLUDED with diversity = -1.0 (adjusted) or 0.0 (Shannon)

### Data Quality Issues
- Some animals may appear in diversity CSV files but have no actual cells for that region/celltype
- These are identified by checking the original h5ad files
- Plotting scripts should filter to only include animals with cells (see `replot_specific_alpha_FINAL.py` for correct implementation)

### Shannon Diversity Special Case
- When an animal has only 1 subtype, Shannon diversity may be stored as NaN in the CSV
- This NaN should be replaced with 0.0 (valid biological data) when the animal has cells
- If the animal has NO cells, exclude entirely

## Scripts

### Core Analysis Pipeline

#### `calc_diversity.py`
Calculate alpha and beta diversity metrics.
```bash
# Cell type level
./calc_diversity.py --region HIP --level celltype

# Subtype level for specific cell type
./calc_diversity.py --region HIP --level subtype --celltype "GABAergic neurons"
```

**Outputs:**
- `{region}_{level}_alpha_diversity_adjusted.csv` - Adjusted entropy values
- `{region}_{level}_alpha_diversity_shannon.csv` - Shannon diversity values
- `{region}_{level}_beta_diversity.csv` - Pairwise Bray-Curtis distance matrix
- `{region}_{level}_metadata.csv` - Animal age information

#### `create_min_beta_csvs.py`
Generate minimum beta diversity (uniqueness) metrics from pairwise distance matrices.
```bash
./create_min_beta_csvs.py
```

**Outputs:**
- `{region}_{level}_min_beta_diversity.csv` - Minimum distance to nearest neighbor for each animal

**Note**: Automatically excludes animals with no valid neighbors (isolated samples)

### Statistical Analysis

#### `test_age_diversity.py`
Test linear relationships between age and diversity metrics.
```bash
./test_age_diversity.py --region HIP --level subtype
```

**Outputs:**
- `{region}_{level}_age_effects_adjusted.csv` - Age correlations for adjusted entropy
- `{region}_{level}_age_effects_shannon.csv` - Age correlations for Shannon diversity
- `{region}_{level}_age_effects_min_beta.csv` - Age correlations for min beta diversity

### Visualization Scripts

#### Individual Regression Plots
```bash
# Alpha diversity (adjusted entropy)
./plot_alpha_regression.py --region HIP --celltype "GABAergic neurons"

# Beta diversity (mean)
./plot_beta_regression.py --region HIP --celltype "microglia"

# Beta diversity (minimum/uniqueness)
./plot_min_beta_regression.py --region HIP --celltype "microglia"
```

#### Summary Visualizations
```bash
# Four-panel forest plots (all metrics)
./plot_four_panel_forest.py --region HIP --level subtype

# MDS ordination plots
./create_mds_plots_mean_and_min.py  # Creates MDS for both mean and min beta

# Composition bar plots
./plot_composition_bars.py --region HIP --level subtype
```

#### Comparative Analysis
```bash
# Mean vs min beta analysis (distribution comparison)
./analyze_mean_vs_min_beta.py
```

## Typical Workflow

### 1. Calculate diversity metrics
```bash
# For each region and cell type
./calc_diversity.py --region HIP --level celltype
./calc_diversity.py --region HIP --level subtype --celltype "GABAergic neurons"
./calc_diversity.py --region HIP --level subtype --celltype "microglia"
# ... repeat for all regions/celltypes
```

### 2. Generate minimum beta diversity
```bash
./create_min_beta_csvs.py
```

### 3. Test age effects
```bash
./test_age_diversity.py --region HIP --level celltype
./test_age_diversity.py --region HIP --level subtype
```

### 4. Create visualizations
```bash
# Individual regressions
./plot_alpha_regression.py --region HIP --celltype "microglia"

# Summary plots
./plot_four_panel_forest.py --region HIP --level subtype

# MDS ordinations
./create_mds_plots_mean_and_min.py
```

## Output Directory Structure
```
diversity/
  ├── {region}_celltype_alpha_diversity_adjusted.csv
  ├── {region}_celltype_alpha_diversity_shannon.csv
  ├── {region}_celltype_beta_diversity.csv
  ├── {region}_celltype_min_beta_diversity.csv
  ├── {region}_{celltype}_subtype_alpha_diversity_adjusted.csv
  ├── {region}_{celltype}_subtype_alpha_diversity_shannon.csv
  ├── {region}_{celltype}_subtype_beta_diversity.csv
  ├── {region}_{celltype}_subtype_min_beta_diversity.csv
  └── ...

diversity_figs/
  ├── {region}_celltype_alpha_regression_adjusted.png
  ├── {region}_celltype_beta_regression.png
  ├── {region}_celltype_min_beta_regression.png
  ├── {region}_celltype_age_effects_four_panel.png
  ├── {region}_celltype_mds_mean_beta.png
  ├── {region}_celltype_mds_min_beta.png
  └── ...

composition_figs/
  ├── {region}_celltype_composition.png
  ├── {region}_{celltype}_subtype_composition.png
  └── ...
```

## Requirements
```bash
# Core dependencies
pip install scanpy pandas numpy scipy scikit-learn matplotlib seaborn

# For running on cluster
module load anaconda3
conda activate latent_analysis
```

## Key Considerations

### Statistical Testing
- P-values from linear regressions are **uncorrected** for multiple comparisons
- Consider applying FDR correction (Benjamini-Hochberg) when reporting results
- Red bars in forest plots indicate p < 0.05 (nominal significance)

### Data Filtering
When creating plots, ensure proper filtering:
```python
# Correct approach: filter to animals with cells
animals_with_cells = get_animals_with_cells(region, celltype)
alpha_df = alpha_df[alpha_df.index.isin(animals_with_cells)]

# Handle Shannon NaN for one-subtype cases
if metric == 'shannon_diversity':
    alpha_df[metric] = alpha_df[metric].fillna(0.0)
```

### Interpreting Metrics
- **Negative age correlation for alpha**: Diversity decreases with age (cells become more uniform)
- **Positive age correlation for mean beta**: Individuals become more different from each other with age
- **Positive age correlation for min beta**: Individuals become more unique (diverge from nearest neighbor)

## Citation

If using this pipeline, please cite:
- Original data source: [Add citation]
- Analysis methods based on Wilmanski et al. 2021 Nature Metabolism (for min beta/uniqueness metric)

## Contact

For questions or issues, contact: [Add contact info]
