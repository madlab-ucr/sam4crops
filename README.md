# Quantifying SAM's performance on multi-class segmentation using clustering consensus metrics

Poses the problem of quantifying SAM's zero-shot performance on multiclass segmentation as a clustering consensus problem.

**Paper:** https://arxiv.org/pdf/2311.15138.pdf

## Codebase

1. `src`: Folder containing scripts

    - `GettingStarted.ipynb`: My all-in-one notebook for a brief EDA and final results visualization.
    - `grid_search.py`: Script for grid search over all experimental parameters.
    - `utils.py`: Useful plotting and other utils.
    - `make_aoi_samples.py`: Script to make samples for experiments from the CalCrop21 benchmark. 
    - `unsuable.tiles.txt`: This are the tiles from Calcrop21 that are deemed not suitable for this analysis after the max NDVI RGB extraction.
    - `colormap.py`: A colormap for the CDL.

2. `cached_models`: Folder to save SAM weights

3. `results`: Folder to store results



