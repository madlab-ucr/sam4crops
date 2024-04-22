
'''
File Created: Wednesday, 14th June 2023 1:08:40 am
Author: Rutuja Gurav (rgura001@ucr.edu)
'''

'''
clear; python -u -B src/grid_search.py --num_samples 300 --use_gpu 3 --aoi_sizes 1098 &> stdout/gridsearch_$(date "+%Y%m%d%H%M%S").out &
clear; python -u -B src/grid_search.py --num_samples 300 --use_gpu 4 --aoi_sizes 549 &> stdout/gridsearch_$(date "+%Y%m%d%H%M%S").out &
clear; python -u -B src/grid_search.py --num_samples 300 --use_gpu 5 --aoi_sizes 274 &> stdout/gridsearch_$(date "+%Y%m%d%H%M%S").out &
clear; python -u -B src/grid_search.py --num_samples 300 --use_gpu 6 --aoi_sizes 137 &> stdout/gridsearch_$(date "+%Y%m%d%H%M%S").out &

'''
import re
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor']='white' 
import seaborn as sns
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
import numpy as np
import pandas as pd
from sklearn import metrics
from PIL import Image
from tqdm import tqdm
import glob, collections, random, os, datetime, json_tricks, traceback

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from utils import plot_prediction, eval_clustering

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--use_gpu', type=int, default=7,
                    help='an integer indicating which gpu to use')

parser.add_argument('--num_samples', type=int, default=300,
                    help='no. of datapoints to use in experiments')

parser.add_argument('--data_dir', default="/data/rgura001/AI4CP.data/sam4crops/aoi_samples/overlap_0.5/",
                    help='path to aoi samples dir')

parser.add_argument('--model_type', default='vit_h',
                    help='SAM version to use: vit_l, vit_h, vit_b')  

parser.add_argument('--model_chkpt_dir', default="/home/rgura001/segment-anything/sam4crops/cached_models",
                    help='dir where SAM weights are saved in pth files')   

parser.add_argument('--aoi_sizes', metavar='N', type=int, nargs='+', default=[1098, 549, 274, 137],
                    help='')

parser.add_argument('--min_mask_region_fracs_vals', metavar='N', type=float, nargs='+', default=[1e-3, 5*1e-3, 1e-2, 5*1e-2, 1e-1],
                    help='what fraction of the image area should be the minimum mask region area?')

parser.add_argument('--pps_fracs_vals', metavar='N', type=float, nargs='+', default=[1e-2, 2.5e-2, 5e-2, 7.5e-2, 1e-1],
                    help='how many points per side to uniformly generate as fraction of length of image?')

parser.add_argument('--abs_pps_vals', metavar='N', type=int, nargs='+', default=[4, 8, 16, 32],
                    help='how many points per side to uniformly generate?')

parser.add_argument('--crop_n_layers_vals', metavar='N', type=int, nargs='+', 
                    default=[1],
                    help='how many crops to do on the image?')

parser.add_argument('--crop_n_points_downscale_factor_vals', metavar='N', type=int, nargs='+', 
                    default=[1],
                    help='how much to downscale the image after each crop?')

parser.add_argument('--results_file_id', default="",
                    help='identifier to add to results file name')

args = parser.parse_args()

device = "cuda:{}".format(args.use_gpu)
num_samples = args.num_samples

RESULTS_DIR = f"/home/rgura001/segment-anything/sam4crops/results/grid_search/num_samples_{num_samples}"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# RESULTS_DIR_ALT = f"/data/rgura001/segment-anything/sam4crops/results/grid_search/num_samples_{num_samples}"
# if not os.path.exists(RESULTS_DIR_ALT):
#     os.makedirs(RESULTS_DIR_ALT)

model_type = args.model_type
MODEL_DIR = args.model_chkpt_dir
sam_checkpoint = [fp for fp in glob.glob(MODEL_DIR+'/*.pth') if model_type in fp][0]
print(sam_checkpoint)
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device) # type: ignore

aoi_sizes = args.aoi_sizes
min_mask_region_fracs_vals = args.min_mask_region_fracs_vals
pps_fracs_vals = args.pps_fracs_vals
# pps_vals = args.abs_pps_vals
crop_n_layers_vals = args.crop_n_layers_vals
crop_n_points_downscale_factor = 1
nms_thresh = 0.7
# print(aoi_sizes, min_mask_region_fracs, pps_vals)

tic = datetime.datetime.now()
results = pd.DataFrame()
print(results.shape)
for aoi_size in aoi_sizes:
    img_w, img_h = aoi_size, aoi_size
    AOI_SAMPLES_DIR = args.data_dir+"/DATAPOINTS_maxNDVItimestep_rgb_size_{}".format(aoi_size) ## these are all the samples AFTER filtering for unusable tiles.
    sample_filepaths = glob.glob(AOI_SAMPLES_DIR+'/*.npy')
    # print(len(samples_filepaths), samples_filepaths[:2])
    num_avail_samples = len(sample_filepaths)
    
    # if num_avail_samples >= num_samples:
    #     np.random.seed(42) ## Fix this seed for reproducibility
    #     sample_idxes = np.random.randint(low=0, high=num_avail_samples, size=num_samples)
    # else:
    #     print("num_samples greater than num_avail_samples(={})! Exiting...".format(num_avail_samples))
    #     sys.exit()
    # print(len(sample_idxes))

    np.random.seed(42) ## Fix this seed for reproducibility
    all_sample_idxes = np.arange(num_avail_samples)
    np.random.shuffle(all_sample_idxes)
    # sample_idxes = all_sample_idxes[:num_samples]
    # leftover_sample_idxes = all_sample_idxes[num_samples:]

    for min_mask_area_frac in min_mask_region_fracs_vals:
        min_mask_area = np.ceil(min_mask_area_frac*img_w*img_h).astype(int)
        min_mask_area_perc = min_mask_area_frac*100
        # for pps in tqdm(pps_vals, leave=False):
        for pps_frac in pps_fracs_vals:
            pps = np.ceil(pps_frac*img_w).astype(int)
            pps_perc = pps_frac*100
            for crop_n_layers in crop_n_layers_vals:
                # for crop_n_points_downscale_factor in tqdm(args.crop_n_points_downscale_factor_vals, leave=False):
                print(aoi_size, min_mask_area_perc, pps_perc, crop_n_layers)
                try:
                    mask_generator = SamAutomaticMaskGenerator(model=sam,
                                                                points_per_side=pps,
                                                                pred_iou_thresh=0.95,
                                                                stability_score_thresh=0.95,
                                                                box_nms_thresh=nms_thresh,
                                                                crop_nms_thresh=nms_thresh,
                                                                crop_n_layers=crop_n_layers,
                                                                crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                                                                min_mask_region_area=min_mask_area,  # Requires open-cv to run post-processing
                                                            )
                    print("Created mask generator!")

                    SAVE_DIR=RESULTS_DIR+f"/aoi_size_{aoi_size}/mmra_perc_{min_mask_area_perc}/pps_perc_{pps_perc}/cropnlayers_{crop_n_layers}/nms_thresh_{nms_thresh}"
                    if not os.path.exists(SAVE_DIR):
                        os.makedirs(SAVE_DIR+"/plots")
                        os.makedirs(SAVE_DIR+"/preds")
                    
                    # SAVE_DIR_ALT=RESULTS_DIR_ALT+f"/aoi_size_{aoi_size}/mmra_perc_{min_mask_area_perc}/pps_perc_{pps_perc}/cropnlayers_{crop_n_layers}/nms_thresh_{nms_thresh}"
                    # if not os.path.exists(SAVE_DIR_ALT):
                    #     os.makedirs(SAVE_DIR_ALT+"/preds/boolean_masks")

                    result_df = pd.DataFrame()
                    # result_df['sample_idx'] = sample_idxes
                    result_df['aoi'] = [aoi_size]*num_samples
                    result_df['pps'] = [pps]*num_samples
                    result_df['pps_perc'] = [pps_perc]*num_samples
                    result_df['mmra'] = [min_mask_area]*num_samples
                    result_df['mmra_perc'] = [min_mask_area_perc]*num_samples
                    result_df['crop_n_layers'] = [crop_n_layers]*num_samples
                    result_df['crop_n_points_downscale_factor'] = [crop_n_points_downscale_factor]*num_samples
                    result_df['nms_thresh'] = [nms_thresh]*num_samples

                    scores = []
                    sample_idxes = []
                    ok_sample_count = 0
                    # for idx in sample_idxes:
                    for idx in all_sample_idxes:
                        print("Processing {}th sample with id {}...".format(ok_sample_count, idx))
                        sample = np.load(sample_filepaths[idx])   
                        image, gt_mask = sample[...,:3].astype('uint8'), sample[...,-1].astype('uint8')
                        # print(image.shape, gt_mask.shape)
                        print(f"num_labels_in_gt_mask = {len(np.unique(gt_mask))}")
                        
                        try: 
                            masks = mask_generator.generate(image)
                            print("For sample {}, SAM generated {} boolean masks!".format(idx, len(masks)))
                            ok_sample_count += 1
                            sample_idxes.append(idx)
                        except Exception as e:
                            print("Couldn't generate masks for sample {}! Continuing...".format(idx))
                            print(e)
                            print(traceback.format_exc())
                            continue
                        
                        ## Sort masks by stability score
                        masks = sorted(masks, key=lambda d: d['stability_score']) 
                        ## Transform N boolean overlapping masks into one numerically encoded mask
                        pred_mask = np.zeros((image.shape[0], image.shape[1]))
                        # pred_mask = masks[0]['segmentation'].astype(int)
                        for i in range(len(masks)):
                            val = i+1
                            pred_mask += masks[i]['segmentation'].astype(int) * val
                            pred_mask = np.where(pred_mask > val, val, pred_mask)
                        pred_mask = pred_mask.astype('uint8')
                        print(f"Max pred cluster id = {pred_mask.flatten().max()}")
                        print(f"num_labels_in_pred_mask = {len(np.unique(pred_mask))}")
                        # print("DONE!")

                        sample_result_arr = np.concatenate([image, gt_mask[...,np.newaxis], pred_mask[...,np.newaxis]], axis=-1)
                        np.save(SAVE_DIR+'/preds/sample_{}'.format(idx), sample_result_arr)
                        
                        ## Save boolean masks into a json file
                        # with open(SAVE_DIR_ALT+"/preds/boolean_masks/sample_{}.json".format(idx), "w") as fp:
                        #     json_tricks.dump(masks, fp)

                        print("Evaluating clustering consensus...")
                        clust_score = eval_clustering(gt_mask.flatten(), pred_mask.flatten())
                        clust_score['num_labels_in_pred_mask'] = len(np.unique(pred_mask))
                        clust_score['num_labels_in_gt_mask'] = len(np.unique(gt_mask))
                        scores.append(clust_score)

                        titlestr = "AOI=({}x{}) (SAM params: PPS={}% of sqrt(AOI), MMRA={}% of AOI,\ncrop_n_layers={}, nms_threshold={})".\
                                    format(img_w, img_h, pps_perc, min_mask_area_perc, crop_n_layers, nms_thresh)
                        
                        titlestr += f"\n(Metrics: FMI={clust_score['FMI']}, ARI={clust_score['ARI']},\
                                    \nMI={clust_score['MI']}, NMI={clust_score['NMI']}, AMI={clust_score['AMI']},\
                                    \nHomogeneity={clust_score['Homogeneity']}, Completeness={clust_score['Completeness']}, V-Measure={clust_score['V-Measure']})"
                        titlestr = f"Sample #{idx}\n"+titlestr
                        plot_prediction(input=image, gt_mask=gt_mask, pred_mask=pred_mask, 
                                        titlestr=titlestr,
                                        show=False, save=True, SAVE_PATH=SAVE_DIR+"/plots/sample_{}_titled.png".format(idx)
                                    )
                        plt.close()
                        plot_prediction(input=image, gt_mask=gt_mask, pred_mask=pred_mask, 
                                        titlestr="",
                                        show=False, save=True, SAVE_PATH=SAVE_DIR+"/plots/sample_{}.png".format(idx)
                                    )
                        plt.close()

                        if ok_sample_count == num_samples:
                            break

                    result_df.insert(loc=0, column='sample_idx', value=sample_idxes)
                    scores = pd.DataFrame(scores)
                    col_names = result_df.columns.to_list()+scores.columns.to_list()
                    result_df = pd.concat([result_df, scores], ignore_index=True, axis=1)
                    result_df.columns = col_names
                    results = pd.concat([results, result_df], ignore_index=True, axis=0)
                    # results = results.sort_values(by=['sample_idx'])
                    results.to_csv(RESULTS_DIR+f"/aoi_size_{aoi_size}/results{args.results_file_id}.csv", index=False)
                
                except Exception as e:
                    print("Error encountered! Continuing...")
                    print(e)
                    print(traceback.format_exc())
                    continue

    print(results.shape)
    print(collections.Counter(results['pps_perc']))
    print(len(results['sample_idx'].unique()))
    results.to_csv(RESULTS_DIR+f"/aoi_size_{aoi_size}/results.csv", index=False)                        
    print("Total Time taken: {}".format(datetime.datetime.now()-tic))
