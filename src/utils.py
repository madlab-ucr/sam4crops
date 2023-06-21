'''
File Created: Thursday, 15th June 2023 10:41:51 am
Author: Rutuja Gurav (rgura001@ucr.edu)
'''

import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor']='white'
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

import seaborn as sns
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

import numpy as np
import pandas as pd

from colormap import getCDLRGB, getCDLHEX

def colormap_mask(temp):

    label_colors=[np.array(getCDLRGB(i)) for i in range(0,256)]
    label_colors=np.array(label_colors)
    
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0,256):
        r[temp==l]=label_colors[l,0]
        g[temp==l]=label_colors[l,1]
        b[temp==l]=label_colors[l,2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:,:,0] = (r/255.0)#[:,:,0]
    rgb[:,:,1] = (g/255.0)#[:,:,1]
    rgb[:,:,2] = (b/255.0)#[:,:,2]
    return rgb

def plot_sample(img=None, mask=None, save=False, SAVE_PATH=None, show=True, titlestr=""):
    fig, axs = plt.subplots(1,2, figsize=(10,7))
    axs[0].imshow(img)
    axs[0].set_title("Input", fontsize=20)

    class_names = ['Corn','Cotton','Rice','Sunflower','Barley','Winter_Wheat','Safflower','Dry Beans','Onions','Tomatoes','Cherries','Grapes','Citrus','Almonds','Walnut','Pistachio','Garlic','Olives','Pomegranates','Alfalfa','Hay','Barren_land','Fallow_and_Idle','Forests_combined','Grass_combined','Wetlands_combined','Water','Urban']
    labels_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28, 100]
    cdl_values_list = [0, 1, 2, 3, 6, 21, 24, 33, 42, 49, 54, 66, 69, 72, 75, 76, 204, 208, 211, 217, 36, 37, 65, 61, 63, 176, 87, 83, 82, 100]
    labels_cdl_map = dict(zip(labels_list, cdl_values_list))

    mask = colormap_mask(np.vectorize(labels_cdl_map.get)(mask))
    
    axs[1].imshow(mask)
    axs[1].set_title("Ground Truth", fontsize=20)

    axs[0].axis('off')
    axs[1].axis('off')

    plt.suptitle(titlestr, fontsize=20)

    if show:
        plt.show()
    if save:
        plt.savefig(SAVE_PATH)
    plt.close()

def plot_prediction(input=None, gt_mask=None, pred_mask=None, 
                    titlestr="", show=False, save=False, SAVE_PATH=None):
    fig, axs = plt.subplots(1,3, figsize=(10,7))
    axs[0].imshow(input)
    axs[0].set_title("Input", fontsize=20)

    # class_names = ['Corn','Cotton','Rice','Sunflower','Barley','Winter_Wheat','Safflower','Dry Beans','Onions','Tomatoes','Cherries','Grapes','Citrus','Almonds','Walnut','Pistachio','Garlic','Olives','Pomegranates','Alfalfa','Hay','Barren_land','Fallow_and_Idle','Forests_combined','Grass_combined','Wetlands_combined','Water','Urban']
    class_names = ['Background','Corn','Cotton','Rice','Sunflower','Barley','Winter Wheat','Safflower','Dry Beans','Onions','Tomatoes','Cherries','Grapes','Citrus','Almonds','Walnuts','Pistachios','Garlic','Olives','Pomegranates','Alfalfa','Other Hay/Non Alfalfa','Barren','Fallow/Idle Cropland','Forest','Grassland/Pasture','Wetlands','Water','Developed']
    labels_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28, 100]
    cdl_values_list = [0, 1, 2, 3, 6, 21, 24, 33, 42, 49, 54, 66, 69, 72, 75, 76, 204, 208, 211, 217, 36, 37, 65, 61, 63, 176, 87, 83, 82, 100]
    labels_cdl_map = dict(zip(labels_list, cdl_values_list))

    gt_mask = colormap_mask(np.vectorize(labels_cdl_map.get)(gt_mask))
    
    axs[1].imshow(gt_mask)
    axs[1].set_title("Ground Truth", fontsize=20)

    axs[2].imshow(pred_mask, cmap='nipy_spectral')
    axs[2].set_title("Prediction", fontsize=20)

    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')

    plt.suptitle(titlestr, y=0.85)

    if show:
        plt.show()
    if save:
        plt.savefig(SAVE_PATH)
    plt.close()

