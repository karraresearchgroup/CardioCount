import os

def make_file_list(data_dir, postfix='jpg', must_have=None, must_not_have=None, pairs=False):
    """
    Generate a file list of FULL PATH of files of JPGs
    :param: must_have: The file must have this this string to be made into the list
    :param: pairs: If true, output pairs of image name and label name (If patches, default true)
    """
    # If "patches" in the data_dir name, then default we are using pairs
    if 'patches' in data_dir:
        pairs = True

    save_file = os.path.join(data_dir, 'file_list_raw.txt')
    # Clear the previous file
    if os.path.isfile(save_file):
        os.remove(save_file)
    if must_have == 'BW' and pairs:
        save_file = save_file.replace('raw', 'valid')
    elif must_not_have == 'BW' and pairs:
        save_file = save_file.replace('raw', 'train')

    with open(save_file, 'a') as f:
        for files in os.listdir(data_dir):
            if must_have is not None and must_have not in files:
                print('{} does not have must_have component {}, skipping'.format(files, must_have))
                continue
            if must_not_have is not None and must_not_have in files:
                print('{} does not have must_not_have component {}, skipping'.format(files, must_have))
                continue
            if postfix is 'jpg':
                if files.endswith('.JPG') or files.endswith('.jpg'):
                    if pairs:
                        f.write(files)
                        f.write(' ')
                        f.write(files.replace('.jpg', '.png'))
                    else:
                        f.write(os.path.join(data_dir, files))
                    f.write('\n')
            elif postfix is 'png':
                if files.endswith('.PNG') or files.endswith('.png'):
                    f.write(os.path.join(data_dir, files))
                    f.write('\n')
            elif postfix is 'tif':
                if files.endswith('.tif') or files.endswith('.TIF'):
                    f.write(os.path.join(data_dir, files))
                    f.write('\n')


def group_make_file_list(dir_group, postfix='jpg', must_have=None, must_not_have=None, pairs=False):
    """
    Make file list for a group of folders
    """
    for data_dir in dir_group:
        print('Making file list in {}'.format(data_dir))
        make_file_list(data_dir, postfix=postfix, must_have=must_have, must_not_have=must_not_have, pairs=pairs)

if __name__ == '__main__':
    make_file_list('/hpc/group/karralab/test_data/rb_images', postfix='tif') 