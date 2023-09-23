#!/bin/bash

aoi_sizes=(1098 549 274 137)
use_gpu=(4 5 6 7)
num_samples=3
echo "num_samples: $num_samples"

for i in "${!aoi_sizes[@]}"
    do
        aoi_size=${aoi_sizes[$i]}
        use_gpu=${use_gpu[$i]}
        for pps_frac in 0.01 0.025 0.05 0.075 0.1
            do
                for min_mask_region_frac in 0.001 0.005 0.01 0.05 0.1
                    do
                        echo "aoi_size: $aoi_size, use_gpu: $use_gpu, pps_frac: $pps_frac, min_mask_region_frac: $min_mask_region_frac"
                        python -u -B src/grid_search.py --num_samples $num_samples --aoi_sizes $aoi_size --use_gpu $use_gpu --pps_fracs_vals $pps_frac --min_mask_region_fracs_vals $min_mask_region_frac &> stdout/gridsearch_$(date "+%Y%m%d%H%M%S").out &
                        sleep 1
                    done
            done
    done
wait
