from skimage import io
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from mrs_utils.eval_utils import *
from skimage.transform import resize
from shutil import copyfile
from glob import glob 


def post_processing(mother_folder, new_folder_name, csv_name, min_region=50, min_th=0.5, 
                    dilation_size=0, link_r=0, eps=2, 
                    operating_object_confidence_thres=0.1):
    # Loop over all the prediction maps
    for conf_map_name in tqdm(os.listdir(mother_folder)):
        # Skip and only work on the conf_map
        if 'conf.png' not in conf_map_name:
            continue
        print('post processing {}'.format(conf_map_name))
        # Read this conf map
        conf_map = io.imread(os.path.join(mother_folder, conf_map_name))
        # Rescale to 0, 1 interval
        if np.max(conf_map) > 1:
            conf_map  = conf_map / 255
        # Create the object scorer to do the post processing
        obj_scorer = ObjectScorer(min_region, min_th, dilation_size, link_r, eps)
        # Get the object groups after post processing
        group_conf, pixel_grouped = obj_scorer.get_object_groups(conf_map)

        # Loop over each individual groups
        for g_pred in group_conf:
            # Assigning flags for whether to keep this group
            g_pred.keep = True
            # Calculate the average confidence value (to be thresholded)
            _, conf = get_stats_from_group(g_pred, conf_map)
            # IF this object is smaller than what it should be
            if conf < operating_object_confidence_thres:
                # Throw this group away
                g_pred.keep = False
        # Threshold the objects by their confidence values
        group_conf = [a for a in group_conf if a.keep == True]
        print(group_conf)

        cell_number = len(group_conf)

        # dummyfy the result into a binary plot
        conf_dummy = dummyfy(conf_map, group_conf)
        new_folder = os.path.join(mother_folder, '..', new_folder_name)
        if not os.path.isdir(new_folder):
            os.makedirs(new_folder)  
        io.imsave(os.path.join(new_folder, conf_map_name.replace('conf', 'conf_post_processed')), conf_dummy)
        # Outputing the groups
        group_img = conf_dummy * pixel_grouped
        io.imsave(os.path.join(new_folder, conf_map_name.replace('conf', 'conf_grouped')), group_img)

        csv_path = os.path.join(mother_folder, '..', csv_name)
        myCsvRow = ",".join([str(f) for f in [conf_map_name, cell_number]])
        with open(csv_path,'a') as fd:
            fd.write('\n' + myCsvRow)

if __name__ == '__main__':
    base_path = r'/hpc/group/karralab/test_data/'
    input_folder_name = 'probmap_erg'
    new_folder_name = "post_processed_erg"
    csv_name = 'post_processed_erg_stats.csv'
    csv_path = os.path.join(base_path, csv_name)
    operating_object_confidence_threshold = 0.9131550802139036

    myCsvRow = ",".join(['image_name', 'single_pos_count'])
    with open(csv_path,'a') as fd:
        fd.write('\n' + myCsvRow)

    post_processing(os.path.join(base_path, input_folder_name), new_folder_name, csv_name, operating_object_confidence_thres=operating_object_confidence_threshold) 
    