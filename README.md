# Quantifying SAM's performance on multi-class segmentation using clustering consensus metrics

Poses the problem of quantifying SAM's zero-shot performance on multiclass segmentation as a clustering consensus problem.

**Paper:** https://arxiv.org/pdf/2311.15138.pdf

## Setup
1. Get the codebase of SAM - `git clone https://github.com/facebookresearch/segment-anything`
2. Get this codebase and save it in the top-level directory of SAM - `cd segment-anything` then `git clone https://github.com/madlab-ucr/sam4crops.git`
3. Download SAM weights from step 1 repo github page and store them in `segment-anything/sam4crops/cached_models`

## Codebase

1. `src`: Folder containing scripts

    - `GettingStarted.ipynb`: My all-in-one notebook for a brief EDA and prediction visualization.
    - `make_aoi_samples.py`: Script to make samples for experiments from the CalCrop21 benchmark. Step 1 of 3.
    - `grid_search.py`: Script for grid search over all experimental parameters. Step 2 of 3.
    - `ResultsViz.ipynb`: Notebook to visualize results of grid search. Step 3 of 3.
    - `utils.py`: Useful plotting and other utils.
    - `unsuable_tiles.txt`: This are the tiles from Calcrop21 that are deemed not suitable for this analysis after the max NDVI RGB extraction.
    - `colormap.py`: A colormap for the CDL.

2. `cached_models`: Folder to save SAM weights

3. `results`: Folder to store results



