# Import necessary modules
import os
from tqdm import tqdm
from mrs_utils import misc_utils
from data import data_utils

# Set base directory and folders to work with
base_folder = r''   # Insert the path of your base directory
uncropped_image_folder = 'images_rb1'   # Specify the folder name that contains the original images
destination_image_folder = 'images_rb_cropped'   # Specify the folder name where you want to save the cropped images

# Set the source and target paths
source_path = os.path.join(base_folder, uncropped_image_folder)
target_path = os.path.join(base_folder, destination_image_folder)

# Check if the target folder exists, and create it if it doesn't
if not os.path.exists(target_path):
    os.makedirs(target_path)

# Set patch size, overlap, and padding
patch_size = 512 
patch_size=(patch_size,patch_size)
overlap = 0
padding = 0 

# Loop through each image file in the source folder
for img_file in tqdm(os.listdir(source_path)):
  
  # Get the file name and prefix
  img_file = os.path.join(source_path, img_file)
  prefix = os.path.splitext((os.path.basename(img_file)))[0]
  
  # Load the image file and make a grid of patches
  rgb = misc_utils.load_file(img_file)
  grid_list = data_utils.make_grid(np.array(rgb.shape[:2]) + 2 * padding, patch_size, overlap)
  
  # Loop through each patch and save it as a new image file
  for y, x in grid_list:
    rgb_patch = data_utils.crop_image(rgb, y, x, patch_size[0], patch_size[1])
    img_patchname = '{}_y{}x{}.jpg'.format(prefix, int(y), int(x))
    misc_utils.save_file(os.path.join(target_path, img_patchname), rgb_patch.astype(np.uint8))
