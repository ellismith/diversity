#!/bin/bash

# All regions including HIP and CN
REGIONS=("HIP" "CN" "ACC" "dlPFC" "EC" "IPP" "lCb" "M1" "MB" "mdTN" "NAc")

echo "Creating individual regression plots for all regions..."

TOTAL=${#REGIONS[@]}
CURRENT=0

for REGION in "${REGIONS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo "[$CURRENT/$TOTAL] Processing $REGION..."
    
    # Plot individual regressions for subtype level
    if [ -f "diversity/${REGION}_subtype_age_effects_adjusted.csv" ]; then
        ./plot_age_effects.py --stats_file diversity/${REGION}_subtype_age_effects_adjusted.csv --plot_individual
    fi
    
    # Plot for cell type level
    if [ -f "diversity/${REGION}_celltype_age_effects_adjusted.csv" ]; then
        ./plot_age_effects.py --stats_file diversity/${REGION}_celltype_age_effects_adjusted.csv --plot_individual
    fi
done

echo "✓ All regression plots complete!"
