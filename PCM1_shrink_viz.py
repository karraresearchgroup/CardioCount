#!/usr/bin/env python
# coding: utf-8

import time
import os
from skimage import io
import numpy as np
import skimage.measure

def output_analysis(pcm1_image, pcm1_ki67_image, img_name, dest):
    
    label_image = io.imread(pcm1_image)
    input_props = skimage.measure.regionprops(
                label_image, intensity_image=None, cache=True
            )
    input_centroids = [np.int_(obj["centroid"]) for obj in input_props]
    output_segmented = np.zeros_like(label_image)
    for ind, arr in enumerate(input_centroids):
        output_segmented[tuple(arr)] = ind + 1

    pcm1ki67_image = io.imread(pcm1_ki67_image)
    doubleposcount = 0
    for j in range(1,np.max(pcm1ki67_image)+1):
        selected_image = (pcm1ki67_image == j)
        if np.sum(output_segmented * selected_image) == 0:
            pcm1ki67_image = pcm1ki67_image * ~selected_image
        else:
            doubleposcount += 1
    output = pcm1ki67_image

    output[output > 0] = 255
    io.imsave(os.path.join(dest, os.path.splitext(img_name)[0]+'.png'), output)

    csv_path = os.path.join(dest, '..', 'doublepos_postprocessed.csv')
    myCsvRow = ",".join([str(f) for f in [img_name, doubleposcount]])
    with open(csv_path,'a') as fd:
        fd.write('\n' + myCsvRow)

if __name__ == '__main__':
    base_path = '/hpc/group/karralab/test_data'

    image_path = os.path.join(base_path, 'rgb_images')
    pcm1_path = os.path.join(base_path, 'post_processed_pcm1')
    pcm1ki67_path = os.path.join(base_path, 'post_processed_ki67')
    dest = os.path.join(base_path, 'post_processed_double_pos_pcm1_ki67')
    csv_path = os.path.join(image_path, '..', 'doublepos_postprocessed.csv')

    myCsvRow = ",".join(['image_name', 'double_pos_count'])
    with open(csv_path,'a') as fd:
        fd.write('\n' + myCsvRow)

    if not os.path.exists(dest):
        os.makedirs(dest)

    img_names = sorted(os.listdir(image_path))

    img_num= 0
    for f in sorted(os.listdir(pcm1_path)):
        if "grouped" in f:
            pcm1_image = os.path.join(pcm1_path, f)
            pcm1_ki67_image = os.path.join(pcm1ki67_path, f)
            output_analysis(pcm1_image, pcm1_ki67_image, img_names[img_num], dest)
            img_num += 1
  
    