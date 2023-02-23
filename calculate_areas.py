#import matplotlib.pyplot as plt
from skimage import io
import os
import numpy as np
from multiprocessing import Pool
#!pip install tifffile
#!pip install imagecodecs
import tifffile
import imagecodecs
from skimage.filters import threshold_mean
import skimage.morphology as mor
from mrs_utils import misc_utils

def calc_areas(img_name, input_path):
    processed = io.imread(os.path.join(input_path, 'rgb_images', img_name))

    intensity_thres = 15 #The intensity threshold
    empty_threshold = 0.97 #The empty threshold
    count_img = np.reshape(processed, [-1, 3]) #Reshape the imagery into single list
    count_empty = np.sum(np.all(count_img < intensity_thres, axis=1)) #Count the number of empty pixels
    empty_percent = count_empty/512/512
    if empty_percent < empty_threshold: #Only normalize the brightness if there are less than 95% of empty pixels
        pct = 95 #Set the percentile to be normalized
        non_empty = np.ravel(np.mean(processed, axis=-1))  #Reshape into single list, and then 
        non_empty = non_empty[non_empty>intensity_thres]
        per = np.percentile(non_empty, pct) #Get the percentile
        
        processed = processed.astype('float') #Get into float range to process
        processed = np.ravel(processed) #Get into a list
        processed[processed<intensity_thres] = 0 #Get everything smaller than intensity to 0
        processed = np.reshape(processed, [512, 512, -1]) #Get the shape back to a image
        processed *= 255/per #Normalization
        processed = np.minimum(255, processed) #Threshold it
        processed = processed.astype('int') #Get back to integer values

    intensity_threshold = 70 #Threshold the intensity by intensity_threshold
    processed = np.any(processed > intensity_threshold, axis=-1)
    processed = processed.astype('int')

    dilation_size = 20 #Dilation
    processed = mor.dilation(processed, mor.square(dilation_size))

    processed = mor.area_closing(processed, area_threshold=35000)#Close the imagery (get rid of the purple dots in yellow space)

    erosion_size = 20 #Erosion
    processed = mor.binary_erosion(processed, mor.square(erosion_size))

    processed = mor.area_opening(processed, area_threshold=100) #Open the imagery (get rid of yellow spots in purple space)

    dilation_size = 40 #Dilation
    processed = mor.binary_dilation(processed, mor.square(dilation_size))

    erosion_size = 40 #Erosion
    processed = mor.binary_erosion(processed, mor.square(erosion_size))

    processed = mor.area_opening(processed, area_threshold=2000) #Open the imagery (get rid of yellow in purple)

    processed = mor.area_closing(processed, area_threshold=10000) #Close the imagery (get rid of purple in yellow)

    test = np.sum(processed)
    tissue_frac = ((test/(512*512)))


    csv_path = os.path.join(input_path, 'areastats.csv')
    myCsvRow = ",".join([str(f) for f in [img_name, test, tissue_frac]])
    with open(csv_path,'a') as fd:
        fd.write('\n' + myCsvRow)
    output_im_path = os.path.join(input_path, 'output_area_thresh', os.path.splitext(img_name)[0]+'.png')
    misc_utils.save_file(output_im_path, processed)

    

if __name__ == '__main__':
    input_path = os.path.join(r'/hpc/group/karralab/test_data/rgb_images')

    csv_path = os.path.join(input_path, 'areastats.csv')
    myCsvRow = ",".join(['image_name', 'pixel_count', 'area'])
    with open(csv_path,'a') as fd:
        fd.write('\n' + myCsvRow)

    output_im_path = os.path.join(input_path, 'output_area_thresh')
    if not os.path.exists(output_im_path):
        os.makedirs(output_im_path)

    num_cpu = 64
    try:
        pool = Pool(num_cpu)
        args_list = []
        for img in os.listdir(os.path.join(input_path, 'rgb_images')):
            if (img.endswith(".tif")):
                args_list.append((img, input_path))
        print(args_list)
        pool.starmap(calc_areas, args_list)
    finally:
        pool.close()
        pool.join()