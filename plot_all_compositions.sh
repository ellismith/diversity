#!/bin/bash

regions=(ACC CN dlPFC EC HIP IPP lCb M1 MB mdTN NAc)

for region in "${regions[@]}"; do
    echo "Processing $region..."
    python plot_cell_composition.py \
        --region_file /scratch/nsnyderm/u01/intermediate_files/regions_h5ad_update/${region}.h5ad
done

echo "Done! All composition plots saved to /scratch/easmit31/variability/composition_figs"
