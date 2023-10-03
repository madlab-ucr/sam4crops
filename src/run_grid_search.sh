#!/bin/bash

aoi_size=1098
use_gpu=(2 3 4 5 6)
min_mask_region_fracs=(0.001 0.005 0.01 0.05 0.1)
num_samples=300
echo "num_samples: $num_samples"

for i in "${!min_mask_region_fracs[@]}"
    do
        min_mask_region_frac=${min_mask_region_fracs[$i]}
        use_gpu=${use_gpu[$i]}
        echo "num_samples: $num_samples, aoi_size: $aoi_size, use_gpu: $use_gpu, min_mask_region_frac: $min_mask_region_frac"
        python -u -B src/grid_search.py --results_file_id $min_mask_region_frac --aoi_sizes $aoi_size --min_mask_region_fracs_vals $min_mask_region_frac --num_samples $num_samples --use_gpu $use_gpu &> stdout/gridsearch_$(date "+%Y%m%d%H%M%S").out &
        sleep 1
    done
wait
