# Cell Type Diversity Analysis

Analysis pipeline for calculating and visualizing alpha and beta diversity of cell types and subtypes across brain regions in aging primates.

## Scripts

### Diversity Calculations
- `calc_diversity.py` - Calculate alpha (Shannon) and beta (Bray-Curtis) diversity
- `test_age_diversity.py` - Test linear relationships between age and diversity metrics

### Visualization
- `plot_diversity.py` - Plot individual diversity metrics (scatter + MDS)
- `plot_composition_bars.py` - Stacked bar plots of cell type/subtype compositions
- `plot_diversity_summary.py` - Violin plots comparing diversity across cell types
- `plot_age_effects.py` - Regression plots and forest plots of age effects

## Usage

### Calculate diversity for a region
```bash
# Cell type level
./calc_diversity.py --region HIP --level celltype

# Subtype level for specific cell type
./calc_diversity.py --region HIP --level subtype --celltype "GABAergic neurons"
```

### Test age effects
```bash
./test_age_diversity.py --region HIP --level subtype
```

### Visualizations
```bash
# Individual plots
./plot_diversity.py --alpha_file diversity/HIP_GABAergic_neurons_subtype_alpha_diversity.csv \
                    --beta_file diversity/HIP_GABAergic_neurons_subtype_beta_diversity.csv \
                    --title_prefix "HIP GABAergic"

# Summary comparisons
./plot_diversity_summary.py --region HIP --level subtype

# Age effect visualizations
./plot_age_effects.py --stats_file diversity/HIP_subtype_age_effects.csv --plot_individual
```

## Requirements
- Python 3.x
- scanpy
- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- seaborn
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
# Data files
*.csv
*.h5ad

# Output directories
diversity/
diversity_figs/
composition_figs/
entropy_csvs/
entropy_figs/
pc_variability/

# Python
__pycache__/
*.pyc
*.pyo

# System
.DS_Store
EOF
