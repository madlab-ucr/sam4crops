import numpy as np
import pandas as pd
import glob, os
import skimage
from scipy import ndimage
from tqdm import tqdm
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor

DATA_DIR = "/data/hpate061/CalCROP21/ACCEPTABLE_GRIDS"
filepaths = glob.glob(DATA_DIR+"/IMAGE_GRIDS/*.npy")
# print(len(filepaths))
# print(filepaths[:2])
tile_ids = [fp.split('/')[-1].rsplit('_',1)[0] for fp in filepaths]
# print(tile_ids[:2])

with open("/home/rgura001/segment-anything/sam4crops/unusable_tiles.txt", 'r') as file:
    unusable_tiles = [line.rstrip() for line in file]

NDVI_SAVE_DIR = "/data/hpate061/CalCROP21/ACCEPTABLE_GRIDS/NDVI_CUBES"
PHENOFEATS_SAVE_DIR = "/data/hpate061/CalCROP21/ACCEPTABLE_GRIDS/PHENOLOGY_FEATURES_CUBES"
os.makedirs(NDVI_SAVE_DIR, exist_ok=True)
os.makedirs(PHENOFEATS_SAVE_DIR, exist_ok=True)

def extract_pheno_features(i, id_):
    image_multispec = np.load(DATA_DIR+"/IMAGE_GRIDS/{}_IMAGE.npy".format(id_))
    gt_mask = np.load(DATA_DIR+"/PREPROCESSED_CDL_GRIDS/{}_PREPROCESSED_CDL_LABEL.npy".format(id_))
    # print(image_multispec.shape, gt_mask.shape)

    B4, B8 = image_multispec[:,2,:,:], image_multispec[:,6,:,:]
    image_ndvi = np.divide((B8-B4), (B8+B4), out=np.zeros_like(B8), where=B4!=0)

    #_________________________________________________________________________________________________________________________________#
    ## Preprocess image_ndvi like interpolation / smoothing of each pixel's timeseries, etc
    image_ndvi_processed = np.empty_like(image_ndvi) ## Stores the processed NDVI cube

    # TODO

    ## Save the processed NDVI cube
    np.save(NDVI_SAVE_DIR+"/{}_NDVI.npy".format(id_), image_ndvi_processed)

    #_________________________________________________________________________________________________________________________________#
    ## Extract phenology features for each pixel's timeseries using Phenolopy - https://github.com/lewistrotter/PhenoloPy
    num_feats = 14 ## Number of phenology features to extract, there are 14 _values features in PhenoloPy, 18 if you include the _times features.
    phenology_features = np.empty((image_ndvi_processed.shape[0], image_ndvi_processed.shape[1], num_feats)) ## Stores the phenology features for the processed NDVI cube
    
    # TODO

    ## Save the phenology features cube
    np.save(PHENOFEATS_SAVE_DIR+"/{}_PHENOLOGY_FEATURES.npy".format(id_), phenology_features)

    #_________________________________________________________________________________________________________________________________#

results = Parallel(n_jobs=-1, verbose=1)(delayed(extract_pheno_features)(i, _id) for i, _id in tqdm(enumerate(tile_ids)))
# print(results)
get_reusable_executor().shutdown(wait=True)
