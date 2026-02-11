#!/bin/bash

# List of remaining regions (already did HIP and CN)
REGIONS=("ACC" "dlPFC" "EC" "IPP" "lCb" "M1" "MB" "mdTN" "NAc")

# Cell types to analyze
CELLTYPES=("GABAergic neurons" "glutamatergic neurons" "astrocytes" "oligodendrocytes" "microglia" "oligodendrocyte precursor cells" "vascular cells")

# Count total tasks for progress
TOTAL_REGIONS=${#REGIONS[@]}
CURRENT=0

for REGION in "${REGIONS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo "=========================================="
    echo "[$CURRENT/$TOTAL_REGIONS] Processing region: $REGION"
    echo "=========================================="
    
    # Cell type level
    echo "[Step 1/5] Calculating cell type diversity..."
    ./calc_diversity.py --region $REGION --level celltype --metrics adjusted
    
    # Subtype level for each cell type (run in parallel)
    echo "[Step 2/5] Calculating subtype diversity (parallel)..."
    for CELLTYPE in "${CELLTYPES[@]}"; do
        ./calc_diversity.py --region $REGION --level subtype --celltype "$CELLTYPE" --metrics adjusted &
    done
    wait  # Wait for all parallel jobs to complete
    
    # Test age effects
    echo "[Step 3/5] Testing age effects..."
    ./test_age_diversity.py --region $REGION --level celltype --metric adjusted
    ./test_age_diversity.py --region $REGION --level subtype --metric adjusted
    
    # Plot cell type level
    echo "[Step 4/5] Plotting diversity..."
    ./plot_diversity.py --alpha_file diversity/${REGION}_celltype_alpha_diversity_adjusted.csv \
                        --beta_file diversity/${REGION}_celltype_beta_diversity.csv \
                        --title_prefix "$REGION Cell Type" &
    
    # Plot subtypes (parallel)
    for CELLTYPE in "${CELLTYPES[@]}"; do
        CELLTYPE_CLEAN=$(echo "$CELLTYPE" | tr ' ' '_' | tr '/' '_')
        
        if [ -f "diversity/${REGION}_${CELLTYPE_CLEAN}_subtype_alpha_diversity_adjusted.csv" ]; then
            ./plot_diversity.py --alpha_file diversity/${REGION}_${CELLTYPE_CLEAN}_subtype_alpha_diversity_adjusted.csv \
                                --beta_file diversity/${REGION}_${CELLTYPE_CLEAN}_subtype_beta_diversity.csv \
                                --title_prefix "$REGION ${CELLTYPE}" &
        fi
    done
    wait  # Wait for all plotting to complete
    
    # Plot composition bars (parallel)
    echo "[Step 5/5] Plotting composition bars..."
    ./plot_composition_bars.py --region $REGION --level celltype &
    
    for CELLTYPE in "${CELLTYPES[@]}"; do
        ./plot_composition_bars.py --region $REGION --level subtype --celltype "$CELLTYPE" &
    done
    wait  # Wait for all composition plots
    
    echo "✓ Completed $REGION ($CURRENT/$TOTAL_REGIONS)"
    echo ""
done

echo "=========================================="
echo "✓ All $TOTAL_REGIONS regions complete!"
echo "=========================================="
