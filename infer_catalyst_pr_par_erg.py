# Built-in
from operator import pos
import os
import argparse
import sys
import shutil

# Libs
import albumentations as A
from albumentations.pytorch import ToTensorV2
from natsort import natsorted
import numpy as np
from skimage import io, measure
from tqdm import tqdm

# Own modules
from mrs_utils import misc_utils, eval_utils
from network import network_io, network_utils
from make_file_list import make_file_list
# Settings

from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from multiprocessing import Pool
import tensorflow as tf


GPU = 0

LOAD_EPOCH = 200
DS_NAME = 'lvad_erg' 
data_specific_folder = r'.'
general_folder = r'/home/home5/agk21/Karra_LVAD_NN/Collins_NeuralNets/Erg_images/Validation_Ims'
DATA_DIR = os.path.join(general_folder, data_specific_folder)  # Parent directory of input images in .jpg format
#SAVE_ROOT = os.path.join(DATA_DIR, 'save_root/') # Parent directory of input images in .jpg format
SAVE_ROOT = os.path.join(r'/home/home5/agk21/Karra_LVAD_NN/Collins_NeuralNets/Erg_images/object_pr_output')
FILE_LIST = os.path.join(DATA_DIR, 'file_list_raw.txt') # A list of full path of images to be tested on in DATA_DIR

PATCH_SIZE = (512, 512)

def calculate_intersection(image_1: np.ndarray, image_2: np.ndarray):
    return np.sum(np.multiply(image_1, image_2))

def calculate_union(image_1: np.ndarray, image_2: np.ndarray):
    return np.sum(np.count_nonzero(image_1 + image_2))

def calculate_iou(image_1: np.ndarray, image_2: np.ndarray):
    intersection = calculate_intersection(image_1, image_2)
    union = calculate_union(image_1, image_2)
    if union == 0:
        return 0
    else:
        return intersection / union

def plain_post_proc(conf, min_conf, min_area):
    tmp = conf > min_conf
    label = measure.label(tmp)
    props = measure.regionprops(label, conf)
    
    dummy = np.zeros(conf.shape)
    for p in props:
        if p.area > min_area:
            for x, y in p.coords:
                dummy[x, y] = 1
    return dummy

def tile_wise_validation(gt_dir, conf_dir, tile_list, min_conf=0.5, min_area=10, 
    progress_bar=False, gt_max=255, conf_max=255, 
    gt_postfix='_GT.png', conf_postfix='_RGB_conf.png', return_arrays=False):

    all_tiles_intersection, all_tiles_union = 0, 0
    tile_iou_dict = {}
    tile_processed_dict = {}
    gt_dict, conf_dict = {}, {}

    tile_iterable = tqdm(tile_list) if progress_bar else tile_list
    for tile_name in tile_iterable:
        gt_dict[tile_name] = io.imread(os.path.join(gt_dir, tile_name+gt_postfix)) / gt_max
        conf_dict[tile_name] = io.imread(os.path.join(conf_dir, tile_name+conf_postfix)) / conf_max
        processed = plain_post_proc(conf_dict[tile_name], min_conf, min_area)
        tile_processed_dict[tile_name] = processed
        
        all_tiles_intersection += calculate_intersection(gt_dict[tile_name], processed)
        all_tiles_union += calculate_union(gt_dict[tile_name], processed)
        tile_iou_dict[tile_name] = calculate_iou(gt_dict[tile_name], processed)

    all_tiles_iou = all_tiles_intersection / all_tiles_union if all_tiles_union != 0 else 0

    if return_arrays:
        return all_tiles_iou, gt_dict, conf_dict, tile_processed_dict
    else:
        return all_tiles_iou


def infer_confidence_map(DATA_DIR=DATA_DIR, SAVE_ROOT=SAVE_ROOT, FILE_LIST=FILE_LIST,
                        DS_NAME=DS_NAME,LOAD_EPOCH=LOAD_EPOCH, extra_save_name=None ):
    """
    Extra save name is for changing the output name of the 
    """
    MODEL_maindir =  r'/usr/project/xtmp/agk21/Karra_LVAD/ergmodels/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=GPU)
    parser.add_argument('--expname', type=str)
    parser.add_argument('--subdir', type=str)
    parser.add_argument('--lossweight', type=str)
    parser.add_argument('--classweight', type=str)
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--save_root', type=str, default=SAVE_ROOT)
    parser.add_argument('--load_epoch', type=int, default=LOAD_EPOCH)
    parser.add_argument('--ds_name', type=str, default=DS_NAME)
    parser.add_argument('--patch_size', type=str, default=PATCH_SIZE)
    parser.add_argument('--file_list', type=str, default=FILE_LIST)
    parser.add_argument('--compute_iou', dest='iou_eval', action='store_true')
    parser.add_argument('--no_compute_iou', dest='iou_eval', action='store_false')
    parser.set_defaults(iou_eval=False)
    super_args = parser.parse_args()
    
    
    exp_name = super_args.expname
    lossweight_name = super_args.lossweight
    classweight_name = super_args.classweight
    
    subfolder_name = super_args.subdir
    MODEL_subdir = os.path.join(MODEL_maindir, subfolder_name)
    MODEL_DIR2 = os.path.join(MODEL_subdir, os.listdir(MODEL_subdir)[0])
    
    log_dir_path = os.path.join(MODEL_DIR2, 'log')
    event_file_path = os.path.join(log_dir_path, os.listdir(log_dir_path)[0])
    for e in tf.compat.v1.train.summary_iterator(event_file_path):
        for v in e.summary.value:
            r = {'metric': v.tag, 'value':v.simple_value}
    training_val_lastiter = r['value']
    
    
    # device, _ = misc_utils.set_gpu(super_args.gpu)
    device = 'cuda:{}'.format(super_args.gpu)


    def load_func_ct_tiles(data_dir, file_list=super_args.file_list, class_names=['panel', ]):
        if file_list:
            if not os.path.isfile(file_list):
                make_file_list(data_dir)
            with open(file_list, 'r') as f:
                rgb_files = f.read().splitlines()
        else:
            from glob import glob
            rgb_files = natsorted(glob(os.path.join(data_dir, '*.jpg')))
        lbl_files = [None] * len(rgb_files)
        assert len(rgb_files) == len(lbl_files)
        return rgb_files, lbl_files

    # make file list for this
    if not os.path.exists(FILE_LIST):
        make_file_list(DATA_DIR)

    # init model
    args = network_io.load_config(MODEL_DIR2)
    model = network_io.create_model(args)
    if LOAD_EPOCH:
        args['trainer']['epochs'] = super_args.load_epoch
    ckpt_dir = os.path.join(
        MODEL_DIR2, 'epoch-{}.pth.tar'.format(args['trainer']['epochs']))
    network_utils.load(model, ckpt_dir)
    print('Loaded from {}'.format(ckpt_dir))
    model.to(device)
    model.eval()

    # eval on dataset
    if os.path.exists(os.path.join('/home/wh145/mrs/data/stats/custom', '{}.npy'.format(DS_NAME))):
        mean, std = np.load(os.path.join(
            '/home/wh145/mrs/data/stats/custom', '{}.npy'.format(DS_NAME)))
        print('Use {} mean and std stats: {}'.format(DS_NAME, (mean, std)))
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        print('Use default (imagenet) mean and std stats: {}'.format((mean, std)))

    tsfm_valid = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    if extra_save_name is None:
        save_dir = os.path.join(super_args.save_root, os.path.join(os.path.basename(
            network_utils.unique_model_name(args)), exp_name), "pred_masks")
    else:
        save_dir = os.path.join(super_args.save_root, extra_save_name)
    evaluator = eval_utils.Evaluator(
        super_args.ds_name, super_args.data_dir, tsfm_valid, device, load_func=load_func_ct_tiles)
    evaluator.infer(model=model, patch_size=PATCH_SIZE, overlap=2*model.lbl_margin,
                    pred_dir=save_dir, save_conf=True)

    if super_args.iou_eval:
        # calculate tile-wise IOU
        with open(super_args.file_list, 'r') as fp:
            tile_list = [os.path.basename(s).split('.')[0] for s in fp.readlines()]

        print(
            tile_wise_validation(
                super_args.data_dir, save_dir, tile_list, min_conf=0.5, min_area=0,
                gt_max=1, conf_max=255, gt_postfix='.png', conf_postfix='_conf.png'
            )
        )
    
    #shutil.rmtree(MODEL_subdir)
    
    return os.path.join(os.path.basename(network_utils.unique_model_name(args)), exp_name), exp_name, lossweight_name, classweight_name, training_val_lastiter

def plot_PR_curve(min_region, dilation_size, link_r, min_th, iou_th, conf_dir_list, tile_name_list, gt_dict, save_title, output_dir, calculate_area=False):
    """
    The funciton to plot the PR curve
    :param min_region: The minimal number of pixels to count
    :param dilation_size: The number of pixels to dialate in image processing
    :param link_r: 
    :param min_th: The minimal threshold to consider as positive prediction
    """
    # Loop through a list of confidence maps
    for conf_dir in tqdm(conf_dir_list):
        plt.figure(figsize=(8, 8))

        # Get the confidence map dictionary where the key is the tile_name and the value is the actual images read from skimage.io
        conf_dict = dict(
            zip(
                tile_name_list,
                [io.imread(os.path.join(conf_dir, f+'_conf.png'))
                for f in tile_name_list]
            )
        )
        # Place holder for confidence list and gt list
        conf_list, true_list = [], []
        area_list = []                                  # Getting the area for the normalized ROC curve
        # Loop over each tiles
        for tile in tqdm(tile_name_list, desc='Tiles'):
            if len(np.shape(gt_dict[tile])) == 3:
                conf_img, lbl_img = conf_dict[tile]/255, gt_dict[tile][:, :, 0]                                 # Get the confidence image and the label image
            else:
                conf_img, lbl_img = conf_dict[tile]/255, gt_dict[tile]  
            # save_confusion_map = conf_dir.split('Catalyst_data')[-1].split('image')[0].replace('/','_') + tile    # THis is for 
            save_confusion_map = conf_dir.split('images/')[-1].replace('/','_') + tile
            # print('the save confusion plot name is :', save_confusion_map)
            conf_tile, true_tile = eval_utils.score(                                                        # Call a function in utils.score to score this
                conf_img, lbl_img, min_region=min_region, min_th=min_th/255, 
                dilation_size=dilation_size, link_r=link_r, iou_th=iou_th)#, save_confusion_map=save_confusion_map)    
            conf_list.extend(conf_tile)
            true_list.extend(true_tile)
        print('number of objects in ground truth = {}'.format(np.sum(true_list)))
        # Plotting the PR curve
        ap, p, r, _ = eval_utils.get_precision_recall(conf_list, true_list) 
        # print('len p = {}, len r = {}, ap = {}'.format(len(p), len(r), ap))
        f1  = 2 * (p * r) / (p + r + 0.000001)
        best_f1_idx = np.argmax(f1[1:]) + 1
        print('best_f1_idx = {}, p = {}, r = {}'.format(best_f1_idx, p[best_f1_idx], r[best_f1_idx]))

        #plt.plot(r[1:], p[1:], label='AP: {:.2f}; Dilation radius: {:.2f}'.format(
        #    ap, dilation_size))

        plt.plot(r[1:], p[1:], label='AP: {:.2f}; Dilation radius: {:.2f}'.format(
            ap, dilation_size))

        plt.plot(r[best_f1_idx], p[best_f1_idx], 'ro')
        plt.annotate(
            'Best F1 point (F1={:.2f})\nPrecision={:.2f}\nRecall={:.2f}'.format(
                f1[best_f1_idx],
                p[best_f1_idx], 
                r[best_f1_idx]
            ),
            (r[best_f1_idx] - 0, p[best_f1_idx] - 0)
        )

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('num_obj_{}'.format(np.sum(true_list))+save_title)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(output_dir, save_title + '.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        # Save the pr values for future plotting
        print('shape of r', np.shape(r))
        print('shape of p', np.shape(p))
        # Saving the pr values
        pr = np.concatenate([np.reshape(r, [-1, 1]), np.reshape(p, [-1, 1])], axis=1)
        print('shape of pr', np.shape(pr))
        np.savetxt(save_path.replace('.png','.txt'), pr)
        # Saving the conf and label list values
        conf_label_pair = np.concatenate([np.reshape(conf_list, [-1, 1]), np.reshape(true_list, [-1, 1])], axis=1)
        np.savetxt(save_path.replace('.png','_conf_label_pair.txt'), conf_label_pair)

        # Plotting the ROC curve
        fpr, tpr, thresholds = metrics.roc_curve(true_list, conf_list, pos_label=1)
        auroc = metrics.auc(fpr, tpr)
        f = plt.figure(figsize=(8,8))
        plt.plot(fpr, tpr,label='AUROC={:.2f}'.format(auroc))
        plt.xlabel('fpr')
        plt.ylabel('tpr')
        plt.legend()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        save_path = os.path.join(output_dir, save_title + 'ROC.png')
        plt.title('ROC_'+save_title)
        plt.savefig(save_path, dpi=300)

        if calculate_area:
            # Plotting the normalized ROC curve
            negative_class_num = np.sum(np.equal(true_list, 0))
            normalizing_factor = negative_class_num / np.sum(area_list)
            normalized_fpr = fpr * normalizing_factor           # The normalized fpr value
            f = plt.figure(figsize=(8,8))
            plt.plot(normalized_fpr, tpr)
            plt.xlabel('normalized_fpr, #/m^2')
            plt.ylabel('tpr')
            plt.ylim([0, 1])
            save_path = os.path.join(output_dir, save_title + 'normalized_ROC.png')
            plt.title('normalized_ROC_'+save_title)
            plt.savefig(save_path, dpi=300)
            nfpr_tpr_pair = np.concatenate([np.reshape(normalized_fpr, [-1, 1]), np.reshape(tpr, [-1, 1]), np.reshape(normalizing_factor * np.ones_like(tpr), [-1, 1])], axis=1)
            np.savetxt(save_path.replace('.png','_nfpr_tpr_pair.txt'), nfpr_tpr_pair)
    
    return f1[best_f1_idx], p[best_f1_idx], r[best_f1_idx], ap, auroc

if __name__ == '__main__':
    folder_name, exp_name, lossweight_name, classweight_name, training_val_lastiter = infer_confidence_map()

    conf_dir_path = os.path.join('/home/home5/agk21/Karra_LVAD_NN/Collins_NeuralNets/Erg_images/object_pr_output/', folder_name, 'pred_masks')
    
    gt_dir = '/home/home5/agk21/Karra_LVAD_NN/Collins_NeuralNets/Erg_images/Validation_Masks'
    
    output_dir = os.path.join('/home/home5/agk21/Karra_LVAD_NN/Collins_NeuralNets/Erg_images/object_pr_output/', folder_name)
    
    conf_dir_list = [conf_dir_path]
    min_region = 50
    dilation_size = 0
    min_th = 0.5
    iou_th = 0.2
    
    conf_dir = conf_dir_list[0]
    tile_name_list = ['_'.join(f.split('_')[:-1]) for f in os.listdir(conf_dir)]
    gt_list = [io.imread(os.path.join(gt_dir, f+'.png')) for f in tile_name_list]
    gt_dict = dict(zip(tile_name_list, gt_list))
    save_title = 'test'
    f1_val, p_val, r_val, ap, auroc =  plot_PR_curve(min_region=min_region, dilation_size=dilation_size, link_r=0, min_th=min_th, 
                                                     iou_th=iou_th, conf_dir_list=conf_dir_list, tile_name_list=tile_name_list,
                                                     gt_dict=gt_dict,save_title=save_title, output_dir=output_dir, calculate_area = False)
    
    csv_path = '/home/home5/agk21/Karra_LVAD_NN/Collins_NeuralNets/Erg_images/object_pr_output/allstats.csv'
    
    myCsvRow = ",".join([str(f) for f in [exp_name, folder_name, lossweight_name, classweight_name, training_val_lastiter, f1_val, p_val, r_val, ap, auroc]])
    with open(csv_path,'a') as fd:
        fd.write('\n' + myCsvRow)
