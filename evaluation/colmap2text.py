from camera_error import write_colmap_to_file
import argparse
import pandas as pd
import numpy as np
from glob import glob
import os
from collections import defaultdict
from tqdm import tqdm

def find_folders(root_dir, name):
    subdirectories = []
    
    # Walk through the directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if any of the subdirectories match the given name
        for dirname in dirnames:
            if name in dirname:
                subdirectories.append(os.path.join(dirpath, dirname))
    
    return subdirectories

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    args = parser.parse_args()

    image_path = args.dir
    colmap_folders = find_folders(image_path, 'colmap')

    for colmap_folder in tqdm(colmap_folders[:]):
        images_bins = [x for x in glob(os.path.join(colmap_folder,'**/sparse/0/images.bin'), recursive=True)]
        for images_bin in images_bins:
            try:
                split_idx = images_bin.find('colmap')
                result_dir = os.path.join(images_bin[:split_idx], 'poses')
                output_path = images_bin[split_idx:].split('/')[1]+'.txt'
                output_path = os.path.join(result_dir, output_path)
                os.makedirs(result_dir, exist_ok=True)
                write_colmap_to_file(images_bin, output_path)
            except Exception as e:
                print('Error', e)
                continue