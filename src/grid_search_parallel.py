print("Unable to paralellelize across GPU devices. Quitting effort.")
import sys
sys.exit()

'''
File Created: Wednesday, 14th June 2023 1:08:40 am
Author: Rutuja Gurav (rgura001@ucr.edu)
'''

'''
Run commamd: 
clear; nohup python -B grid_search_parallel.py --aoi_sizes 1098 549 --num_samples 100 --use_gpu 7 &> stdout/gridsearch_$(date "+%Y%m%d%H%M%S").out &
clear; nohup python -B grid_search_parallel.py --aoi_sizes 549 --num_samples 100 --use_gpu 7 &> stdout/gridsearch_$(date "+%Y%m%d%H%M%S").out &
clear; nohup python -B grid_search_parallel.py --aoi_sizes 274 --num_samples 100 --use_gpu 7 &> stdout/gridsearch_$(date "+%Y%m%d%H%M%S").out &
clear; nohup python -B grid_search_parallel.py --aoi_sizes 137 --num_samples 100 --use_gpu 7 &> stdout/gridsearch_$(date "+%Y%m%d%H%M%S").out &
'''


import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor']='white'
import seaborn as sns
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
import numpy as np
import pandas as pd
from sklearn import metrics
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
import glob, collections, random, os, itertools, math, datetime
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
import torch
torch.cuda.empty_cache()

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from utils import plot_prediction

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--use_gpu', type=int, default=7,
                    help='an integer indicating which gpu to use')

parser.add_argument('--num_samples', type=int, default=100,
                    help='no. of datapoints to use in experiments')

parser.add_argument('--data_dir', default="/data/rgura001/AI4CP/sam4crops/aoi_samples/overlap_0.5/",
                    help='path to oi samples dir')

parser.add_argument('--model_type', default='vit_h',
                    help='SAM version to use: vit_l, vit_h, vit_b')  

parser.add_argument('--model_chkpt_dir', default="/home/rgura001/segment-anything/cached_models",
                    help='dir where SAM weights are saved in pth files')   

parser.add_argument('--aoi_sizes', metavar='N', type=int, nargs='+', default=[1098, 549, 274, 137],
                    help='')

parser.add_argument('--min_mask_region_fracs', metavar='N', type=float, nargs='+', default=[1e-3, 5*1e-3, 1e-2, 5*1e-2],
                    help='')

parser.add_argument('--pps_vals', metavar='N', type=int, nargs='+', default=[4, 8, 16, 32],
                    help='')

args = parser.parse_args()

num_samples = args.num_samples

aoi_sizes = args.aoi_sizes
min_mask_region_fracs = args.min_mask_region_fracs
pps_vals = args.pps_vals

print(aoi_sizes, min_mask_region_fracs, pps_vals)

RESULTS_DIR = "/home/rgura001/segment-anything/sam4crops/results/test/" #{}".format( datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

model_type = args.model_type
MODEL_DIR = args.model_chkpt_dir
sam_checkpoint = [fp for fp in glob.glob(MODEL_DIR+'/*.pth') if model_type in fp][0]
print(sam_checkpoint)
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
device = "cuda:{}".format(args.use_gpu)
sam.to(device=device)

def eval_config(aoi_size, min_mask_region_frac, pps):
    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)
    img_w, img_h = aoi_size, aoi_size
    
    AOI_SAMPLES_DIR = args.data_dir+"/DATAPOINTS_maxNDVItimestep_rgb_size_{}".format(aoi_size)
    sample_filepaths = glob.glob(AOI_SAMPLES_DIR+'/*.npy')
    # print(len(samples_filepaths), samples_filepaths[:2])
    num_avail_samples = len(sample_filepaths)
    if num_avail_samples >= num_samples:
        np.random.seed(42)
        sample_idxes = np.random.randint(low=0, high=num_avail_samples, size=num_samples)
    else:
        print("num_samples greater than num_avail_samples(={})! Exiting...".format(num_avail_samples))
        sys.exit()
    # print(len(sample_idxes))

    min_mask_area = int(min_mask_region_frac*img_w*img_h)
    min_mask_region_perc = min_mask_region_frac*100
    
    SAVE_DIR=RESULTS_DIR+'/grid_search/num_samples_{}/aoi_size_{}/mmra_{}/pps_{}'\
        .format(num_samples, aoi_size, min_mask_region_perc, pps)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR+"/plots")
        os.makedirs(SAVE_DIR+"/preds")

    titlestr = "AOI=({}x{})  (PPS={}, MMRA={}% of AOI)"\
            .format(img_w, img_h, pps, min_mask_region_perc)

    result_df = pd.DataFrame()
    result_df['sample_idx'] = sample_idxes
    result_df['AOI'] = [aoi_size]*num_samples
    result_df['PPS'] = [pps]*num_samples
    result_df['MMRA'] = [min_mask_area]*num_samples
    result_df['perc_MMRA'] = [min_mask_region_perc]*num_samples

    # print(min_mask_area, pps)
    mask_generator = SamAutomaticMaskGenerator(model=sam,
                                                points_per_side=pps,
                                                pred_iou_thresh=0.9,
                                                stability_score_thresh=0.9,
                                                crop_n_layers=1,
                                                crop_n_points_downscale_factor=1,
                                                min_mask_region_area=min_mask_area,  # Requires open-cv to run post-processing
                                            )

    scores = []
    for idx in tqdm(sample_idxes, leave=False):
        sample = np.load(sample_filepaths[idx])   
        image, gt_mask = sample[...,:3].astype('uint8'), sample[...,-1].astype('uint8')
        # print(image.shape, gt_mask.shape)
        
        masks = mask_generator.generate(image)
        # print(len(masks))

        ## How many pixels got no mask?
        # pred_mask = np.logical_or.reduce(np.array([mask['segmentation'] for mask in masks]), axis=0)
        # print(pred_mask.shape)
        # print(collections.Counter(pred_mask.flatten()))

        ## Transform N boolean masks into one numerically encoded mask
        pred_mask = np.zeros((image.shape[0], image.shape[1]))
        for i, mask in enumerate(masks):
            pred_mask += np.where(mask['segmentation'] == True, i+1, 0)
        pred_mask = pred_mask.astype('uint8')
        # print(pred_mask.shape)
        # print(collections.Counter(pred_mask.flatten()))

        sample_result_arr = np.concatenate([image, gt_mask[...,np.newaxis], pred_mask[...,np.newaxis]], axis=-1)
        # print(sample_result_arr.shape)
        np.save(SAVE_DIR+'/preds/sample_{}'.format(idx), sample_result_arr)
        
        fmi = metrics.fowlkes_mallows_score(gt_mask.flatten(), pred_mask.flatten())
        scores.append(fmi)

        plot_prediction(input=image, gt_mask=gt_mask, pred_mask=pred_mask, 
                        titlestr=titlestr+"\nFMI={}".format(fmi.round(3)),
                        show=False, save=True, SAVE_PATH=SAVE_DIR+"/plots/sample_{}.png".format(idx)
                    )
                    
    result_df['FMI'] = scores
    return result_df

# avail_gpus = ["cuda:{}".format(i) for i in range(torch.cuda.device_count())]
# avail_devices = list(range(3,6))
options = list(itertools.product(aoi_sizes, min_mask_region_fracs, pps_vals))
# print(len(options))
# device_order = avail_devices*(int(math.ceil(len(options)/len(avail_devices))))
# device_order = device_order[:len(options)]
# print(len(device_order))

results = Parallel(n_jobs=4, verbose=1)(delayed(eval_config)(aoi_size, min_mask_region_frac, pps) 
            for aoi_size, min_mask_region_frac, pps in tqdm(options))

# print(results)
results = pd.concat(results)
print(results)

results.to_csv(RESULTS_DIR+"/grid_search/num_samples_{}/results.csv".format(num_samples), index=False)
        
get_reusable_executor().shutdown(wait=True)


# for (aoi_size, min_mask_region_frac, pps), device in zip(options, device_order):
#     print(aoi_size, min_mask_region_frac, pps, device)
