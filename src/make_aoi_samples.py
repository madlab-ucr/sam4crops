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

overlap = 0.5
SAVE_DIR = "/data/rgura001/AI4CP.data/sam4crops/aoi_samples/overlap_{}".format(overlap)
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def make_aoi_samples(i, id_):
    image_multispec = np.load(DATA_DIR+"/IMAGE_GRIDS/{}_IMAGE.npy".format(id_))
    gt_mask = np.load(DATA_DIR+"/PREPROCESSED_CDL_GRIDS/{}_PREPROCESSED_CDL_LABEL.npy".format(id_))
    # print(image_multispec.shape, gt_mask.shape)

    B4, B8 = image_multispec[:,2,:,:], image_multispec[:,6,:,:]
    image_ndvi = np.divide((B8-B4), (B8+B4), out=np.zeros_like(B8), where=B4!=0)
    # print(image_ndvi.shape)
    max_ndvi_timestep = np.argmax(np.mean(image_ndvi, axis=(1,2)))
    # sorted_ndvi_timesteps = np.mean(image_ndvi, axis=(1,2)).argsort()[-24:][::-1]
    # print(max_ndvi_timestep)
    image = image_multispec[max_ndvi_timestep,[0,1,2],:,:].T * 255
    image = image.astype(int)
    # print(image.shape, image.min(), image.max())

    image = ndimage.rotate(np.fliplr(image), angle=90).astype('uint8')

    ## Filter out images that are all black like id_ == T10SEH_2018_1_8
    # if not Image.fromarray(image).getbbox(): ## img.getbbox() Returns False if there are no non-black pixels
        # print(id_)
        # return id_
    ## NOTE: Above is not sufficient to filter out all the other kinds of noisy samples. Resorting to manual filtering.

    ##-----------DEBUG--------------
    # plot_sample(img=image, mask=gt_mask, titlestr="Tile ID: {}".format(id_), 
    #             save=True, SAVE_PATH="/home/rgura001/segment-anything/sam4crops/results/eda/tile__{}.png".format(id_),
    #             show=False)
    ##------------------------------

    datapoint = np.concatenate((image, gt_mask[..., np.newaxis]), axis=-1) 
    max_shape = datapoint.shape[0]
    for split_ratio in [2, 4, 8, 16]:
        window_size = (max_shape//split_ratio, max_shape//split_ratio, datapoint.shape[-1])
        # views = np.lib.stride_tricks.sliding_window_view(datapoint, window_size, axis=(0,1))
        views = skimage.util.view_as_windows(datapoint, window_size, step=int(overlap*window_size[0])).squeeze()
        # print(views.shape)
        # np.save(SAVE_DIR+"/views/{}_maxNDVItimestep_rgb_size_{}.npy"
        #         .format(id_, samples.shape[2]), samples)
        samples = views.reshape(-1, views.shape[2], views.shape[3], views.shape[4])
        # print(samples.shape)
        
        aoi_save_dir = SAVE_DIR+"/DATAPOINTS_maxNDVItimestep_rgb_size_{}".format(sample.shape[1])
        if not os.path.exists(aoi_save_dir):
            os.makedirs(aoi_save_dir)
        
        for sample_idx in range(samples.shape[0]):
            sample = samples[sample_idx, ...] #[np.newaxis,...]
            # print(sample.shape)
            np.save(aoi_save_dir+"/{}_sample_{}_maxNDVItimestep_rgb_size_{}.npy"
                .format(id_, sample_idx, sample.shape[1]), sample)

        ##-----------DEBUG----------------------
        # plot_sample(img=sample[...,:3], mask=sample[...,-1], titlestr="Tile ID: {}, Sample #{}".format(id_, sample_idx), 
        #         save=True, SAVE_PATH="/home/rgura001/segment-anything/sam4crops/results/eda/tile__{}.png".format(id_),
        #         show=False)
        ##--------------------------------------
    
    # datapoint = datapoint[np.newaxis,...]
    aoi_save_dir = SAVE_DIR+"/DATAPOINTS_maxNDVItimestep_rgb_size_{}".format(datapoint.shape[1])
    if not os.path.exists(aoi_save_dir):
        os.makedirs(aoi_save_dir)

    np.save(aoi_save_dir+"/{}_sample_{}_maxNDVItimestep_rgb_size_{}.npy".format(id_, i, datapoint.shape[1]), datapoint)
    # print(datapoint.shape)

## Ususable tiles are either all black in RGB or are too contaminated with clouds 
# and errors introduced while removing them...
usable_tile_ids = [_id for _id in tile_ids if _id not in unusable_tiles]
results = Parallel(n_jobs=-1, verbose=1)(delayed(make_aoi_samples)(i, _id) for i, _id in tqdm(enumerate(usable_tile_ids)))
# print(results)
get_reusable_executor().shutdown(wait=True)