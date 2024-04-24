import numpy as np
import pandas as pd
import glob, os
import skimage
import scipy as sp
from scipy import ndimage
import xarray as xr
from multiprocessing import Pool

# Add Phenolopy Scripts folder to path
import sys
sys.path.append('./scripts')
import phenolopy


DATA_DIR = "/data/hpate061/CalCROP21/ACCEPTABLE_GRIDS"
filepaths = glob.glob(DATA_DIR+"/IMAGE_GRIDS/*.npy")
# print(len(filepaths))
# print(filepaths[:2])
tile_ids = [fp.split('/')[-1].rsplit('_',1)[0] for fp in filepaths]
# print(tile_ids[:2])

# with open("/home/rgura001/segment-anything/sam4crops/unusable_tiles.txt", 'r') as file:
#     unusable_tiles = [line.rstrip() for line in file]

NDVI_SAVE_DIR = "/data/hpate061/CalCROP21/ACCEPTABLE_GRIDS/NDVI_CUBES"
PHENOFEATS_SAVE_DIR = "/data/hpate061/CalCROP21/ACCEPTABLE_GRIDS/PHENOLOGY_FEATURES_CUBES"
os.makedirs(NDVI_SAVE_DIR, exist_ok=True)
os.makedirs(PHENOFEATS_SAVE_DIR, exist_ok=True)

def extract_pheno_features(id_):
    image_multispec = np.load(DATA_DIR+"/IMAGE_GRIDS/{}_IMAGE.npy".format(id_))
    gt_mask = np.load(DATA_DIR+"/PREPROCESSED_CDL_GRIDS/{}_PREPROCESSED_CDL_LABEL.npy".format(id_))
    # print(image_multispec.shape, gt_mask.shape)

    B4, B8 = image_multispec[:,2,:,:], image_multispec[:,6,:,:]
    image_ndvi = np.divide((B8-B4), (B8+B4), out=np.zeros_like(B8), where=B4!=0)
    image_ndvi[image_ndvi==0] = np.nan
    
    # Convert to dataarray for interpolation and smoothing
    image_ndvi_da = xr.DataArray(image_ndvi, dims=['time', 'x', 'y'], coords={'time':range(image_ndvi.shape[0])})


    #_________________________________________________________________________________________________________________________________#
    ## Preprocess image_ndvi like interpolation / smoothing of each pixel's timeseries, etc
    image_ndvi_interpolated = image_ndvi_da.interpolate_na(dim="time", method="linear", fill_value="extrapolate")

    ## Smoothing
    window_length = 5
    polyorder = 3
    image_ndvi_smooth = sp.signal.savgol_filter(image_ndvi_interpolated, window_length, polyorder, axis=0)

    ## Save the processed NDVI cube
    print("Saving NDVI cube for tile: ", id_)
    np.save(NDVI_SAVE_DIR+"/{}_NDVI.npy".format(id_), image_ndvi_smooth)

    #_________________________________________________________________________________________________________________________________#
    ## Extract phenology features for each pixel's timeseries using Phenolopy - https://github.com/lewistrotter/PhenoloPy
    ## 
    
    # Convert to xarray dataset
    # sp.signal.savgol_filters returns a numpy array convert back to xarray DataArray
    image_ndvi_smooth = xr.DataArray(image_ndvi_smooth, dims=['time', 'x', 'y'], coords={'time':range(image_ndvi_smooth.shape[0])})
    ds = image_ndvi_smooth.to_dataset(name='veg_index')
    
    ## Add placeholder time information to xarray dataset required by Phenolopy (Only extracting values
        ## Adding time from 2018-01-01 to 2018-12-31 with 14 day intervals
        ## Adding random time of the day to each date 
    base_dates = pd.date_range(start='2018-01-01', periods=24, freq='14D')
    random_times = np.random.randint(0, 50000, size=len(base_dates))  # Generate random seconds
    time_delta = pd.to_timedelta(random_times, unit='s')  # Convert seconds to timedelta
    full_dates = base_dates + time_delta  # Combine the base dates with the random time deltas
    ds['time'] = full_dates
    ds = ds.compute()
    
    ## Extracting phenology features for the datacube
    
    ## Changes made to Phenolopy script
    ## For functions get_mos, get_sos, get_eos, get_roi, and get_rod 
    ## Converted nan values to 0 as follows
    ## In the phenolopy source code right before:
        # Convert Type
            #da_mos_values = da_mos_values.astype('float32')
    ##  # Added: 
            # da_mos_values = da_mos_values.fillna(0)
    ds_phenos = phenolopy.calc_phenometrics(da=ds['veg_index'],
                                        peak_metric='pos', #peack of season
                                        base_metric='vos', # valley of season
                                        method='first_of_slope', # method
                                        factor=0.2,
                                        thresh_sides='two_sided',
                                        abs_value=0.1,
                                        )
    
    
    Phenometrics = ['pos_values',
                'mos_values',  
                'vos_values',
                'bse_values',
                'aos_values',
                'sos_values', 
                'eos_values', 
                'los_values',
                'roi_values', 
                'rod_values', 
                'lios_values',
                'sios_values',
                'liot_values',
                'siot_values'
                ,]
    
    features = [ds_phenos[name].values for name in Phenometrics]
    phenology_features = np.stack(features, axis=0)
    
    print("Saving phenology features for tile: ", id_)
    
    
    ## Save the phenology features cube
    np.save(PHENOFEATS_SAVE_DIR+"/{}_PHENOLOGY_FEATURES.npy".format(id_), phenology_features)
    #_________________________________________________________________________________________________________________________________#

if __name__ == "__main__":
    with Pool(processes=32) as pool:  # Adjust number of processes based on your CPU
        pool.map(extract_pheno_features, tile_ids)
