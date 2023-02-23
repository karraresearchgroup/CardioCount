from multiprocessing import Pool
import os
from mrs_utils import misc_utils
import numpy as np

def remove_channel(img_name, input_path, output_path, channel_to_remove):
    rgb = misc_utils.load_file(os.path.join(input_path,img_name))
    rgb[:,:,channel_to_remove] = 0 
    misc_utils.save_file(os.path.join(output_path, img_name), rgb.astype(np.uint8))

if __name__ == '__main__':
    input_path = r'/hpc/group/karralab/test_data/rgb_images'
    output_path = r'/hpc/group/karralab/test_data/bg_images'
    channel_to_remove = 1 #if 0 remove red, 1 to remove green, 2 to remove blue
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    num_cpu = 64
    try:
        pool = Pool(num_cpu)
        args_list = []
        for img in os.listdir(input_path):
            if (img.endswith(".tif")):
                args_list.append((img, input_path, output_path, channel_to_remove))
        print(args_list)
        pool.starmap(remove_channel, args_list)
    finally:
        pool.close()
        pool.join()