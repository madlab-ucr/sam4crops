'''
File Created: Thursday, 15th June 2023 10:41:51 am
Author: Rutuja Gurav (rgura001@ucr.edu)
'''

import matplotlib.pyplot as plt
from sklearn import metrics
plt.rcParams['figure.facecolor']='white'
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

import seaborn as sns
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

import numpy as np
import pandas as pd

from colormap import getCDLRGB, getCDLHEX


original_class_names = ['Corn','Cotton','Rice','Sunflower','Barley','Winter_Wheat','Safflower','Dry Beans','Onions','Tomatoes','Cherries','Grapes','Citrus','Almonds','Walnut','Pistachio','Garlic','Olives','Pomegranates','Alfalfa','Hay','Barren_land','Fallow_and_Idle','Forests_combined','Grass_combined','Wetlands_combined','Water','Urban']
class_names = ['Background','Corn','Cotton','Rice','Sunflower','Barley','Winter Wheat','Safflower','Dry Beans','Onions','Tomatoes','Cherries','Grapes','Citrus','Almonds','Walnuts','Pistachios','Garlic','Olives','Pomegranates','Alfalfa','Other Hay/Non Alfalfa','Barren','Fallow/Idle Cropland','Forest','Grassland/Pasture','Wetlands','Water','Developed']
labels_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28, 100]
cdl_values_list = [0, 1, 2, 3, 6, 21, 24, 33, 42, 49, 54, 66, 69, 72, 75, 76, 204, 208, 211, 217, 36, 37, 65, 61, 63, 176, 87, 83, 82, 100]
labels_cdl_map = dict(zip(labels_list, cdl_values_list))
labels_class_map = dict(zip(labels_list, class_names))

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
    axs[0].set_title("Input", fontsize=15)

    mask = colormap_mask(np.vectorize(labels_cdl_map.get)(mask))
    
    axs[1].imshow(mask)
    axs[1].set_title("Ground Truth", fontsize=15)

    axs[0].axis('off')
    axs[1].axis('off')

    plt.suptitle(titlestr, fontsize=20, y=0.85)

    if show:
        plt.show()
    if save:
        plt.savefig(SAVE_PATH)
    plt.close()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=50):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='red', marker='*', s=marker_size, 
               edgecolor='white', linewidth = 0.1
               )
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='blue', marker='*', s=marker_size, 
               edgecolor='white', linewidth=0.1)   
    
def plot_bin_prediction(input=None, gt_mask=None, bin_mask=None, bin_mask_cdl=None, pred_mask=None,
                        input_points=None, input_labels=None,
                       titlestr="", show=False, save=False, SAVE_PATH=None):
    cdl_val = np.unique(bin_mask_cdl)[-1]
    fig, axs = plt.subplots(1,4, figsize=(20,10))
    axs[0].imshow(input)
    # show_mask(pred_mask, axs[0])
    show_points(input_points, input_labels, axs[0])
    axs[0].set_title("Input", fontsize=15)

    gt_mask = colormap_mask(np.vectorize(labels_cdl_map.get)(gt_mask))
    axs[1].imshow(gt_mask)
    axs[1].set_title("Multi-class Ground Truth", fontsize=15)

    # bin_mask = colormap_mask(np.vectorize(labels_cdl_map.get)(bin_mask))
    # axs[2].imshow(bin_mask)
    axs[2].imshow(colormap_mask(np.vectorize(labels_cdl_map.get)(bin_mask_cdl)))
    axs[2].set_title("Binary Ground Truth\n(Used for prompts sampling)", fontsize=15)

    axs[3].imshow(colormap_mask(np.vectorize(labels_cdl_map.get)(pred_mask*cdl_val))) # cmap='nipy_spectral'
    axs[3].set_title("Prediction", fontsize=15)

    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')
    axs[3].axis('off')

    plt.suptitle(titlestr, fontsize=20, y=0.95)
    # plt.tight_layout()
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

    gt_mask = colormap_mask(np.vectorize(labels_cdl_map.get)(gt_mask))
    
    axs[1].imshow(gt_mask)
    axs[1].set_title("Ground Truth", fontsize=20)

    axs[2].imshow(pred_mask, cmap='nipy_spectral')
    axs[2].set_title("Prediction", fontsize=20)

    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')

    plt.suptitle(titlestr, y=0.9)
    # plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig(SAVE_PATH)
    plt.close()

def multiclass_iou(true_mask=None, predicted_mask=None):
    """
    Calculate class-wise and mean Intersection over Union (IoU) for segmentation masks.

    Args:
    true_mask (numpy.ndarray): Ground truth segmentation mask with class labels.
    predicted_mask (numpy.ndarray): Predicted segmentation mask with class labels.

    Returns:
    class_iou_scores (dict): A dictionary where keys are class labels and values are IoU scores.
    mean_iou (float): The mean IoU score across all classes.
    """
    unique_labels = np.unique(np.concatenate([true_mask, predicted_mask]))

    class_iou_scores = {}
    intersection_per_class = {}
    union_per_class = {}

    for label in unique_labels:
        # Create binary masks for the specific class in both true and predicted masks
        true_class = (true_mask == label).astype(np.float32)
        predicted_class = (predicted_mask == label).astype(np.float32)

        # Calculate intersection and union for the class
        intersection = np.sum(true_class * predicted_class)
        union = np.sum(true_class) + np.sum(predicted_class) - intersection

        # Calculate IoU for the class
        iou = (intersection + 1e-9) / (union + 1e-9)  # Adding small epsilons to avoid division by zero
        
        # Store the IoU score for the class
        class_iou_scores[label] = iou
        intersection_per_class[label] = intersection
        union_per_class[label] = union

    # Calculate the mean IoU
    mean_iou = np.mean(list(class_iou_scores.values()))

    return class_iou_scores, mean_iou

def multiclass_dice_overlap(true_mask=None, predicted_mask=None):
    """
    Calculate multi-class Dice overlap for semantic segmentation.

    Args:
    ground_truth_mask (numpy.ndarray): The ground truth segmentation mask with class labels.
    predicted_mask (numpy.ndarray): The predicted segmentation mask with class labels.

    Returns:
    class_dice_scores (dict): A dictionary where keys are class labels and values are the Dice coefficients.
    mean_dice_score (float): The mean Dice coefficient across all classes.
    """
    # Get unique class labels from both masks
    unique_labels = np.unique(np.concatenate([true_mask, predicted_mask]))

    class_dice_scores = {}
    
    # Calculate Dice coefficient for each class
    for label in unique_labels:
        # Create binary masks for the specific class in both ground truth and predicted masks
        gt_class = (true_mask == label).astype(np.float32)
        pred_class = (predicted_mask == label).astype(np.float32)

        # Calculate intersection and union
        intersection = np.sum(gt_class * pred_class)
        union = np.sum(gt_class) + np.sum(pred_class)

        # Calculate Dice coefficient for the class
        dice = (2.0 * intersection) / (union + 1e-9)  # Adding a small epsilon to avoid division by zero
        
        # Store the Dice coefficient for the class
        class_dice_scores[label] = dice

    # Calculate the mean Dice coefficient
    mean_dice_score = np.mean(list(class_dice_scores.values()))

    return class_dice_scores, mean_dice_score

def eval_clustering(labels_true, labels_pred):
    # true_labels_entropy, pred_labels_entropy = scipy.stats.entropy(labels_true), scipy.stats.entropy(labels_pred)
    # true_labels_info_gain, pred_labels_info_gain = scipy.stats.entropy(labels_true) / len(set(labels_true)), scipy.stats.entropy(labels_pred) / len(set(labels_pred))
    # kl_div = 1 - scipy.stats.entropy(labels_pred, qk=labels_true)
    # ri = np.round(metrics.rand_score(labels_true, labels_pred), 3)
    ari = np.round(metrics.adjusted_rand_score(labels_true, labels_pred), 3)
    mi = np.round(metrics.mutual_info_score(labels_true, labels_pred), 3)
    ami = np.round(metrics.adjusted_mutual_info_score(labels_true, labels_pred), 3)
    nmi = np.round(metrics.normalized_mutual_info_score(labels_true, labels_pred), 3)
    h = np.round(metrics.homogeneity_score(labels_true, labels_pred), 3)
    c = np.round(metrics.completeness_score(labels_true, labels_pred), 3)
    v = np.round(metrics.v_measure_score(labels_true, labels_pred), 3)
    fmi = np.round(metrics.fowlkes_mallows_score(labels_true, labels_pred), 3)

    return pd.Series([fmi,ari,mi,ami,nmi,h,c,v, 
                    #   true_labels_entropy, pred_labels_entropy, true_labels_info_gain, pred_labels_info_gain, kl_div
                    ], 
                     index=['FMI','ARI','MI','AMI','NMI','Homogeneity','Completeness','V-Measure',
                            # 'true_entropy', 'pred_entropy','true_info_gain', 'pred_info_gain', 'KL_divergence'
                        ])