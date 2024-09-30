from camera_error import calc_camera_error, read_camera_file
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
    parser.add_argument('--gt_path', type=str, default='/sensei-fs/users/scheong/github/Open-Sora/eval/MotionCtrl/camera_poses/eval')
    parser.add_argument('--image_start_idx', type=int, default=0, help="Path to gif files")
    args = parser.parse_args()

    image_path = args.dir
    colmap_folders = find_folders(image_path, 'colmap')
    saved_results = defaultdict(list)

    for colmap_folder in tqdm(colmap_folders[:]):
        #camera_name = colmap_folder.split('/')[-2]
        #camera_gt = os.path.join(args.gt_path, f'{camera_name}.txt')
        images_bins = [x for x in glob(os.path.join(colmap_folder,'**/sparse/0/images.bin'), recursive=True)]
        for images_bin in images_bins:
            try:
                names = images_bin.split('/')
                camera_name = names[-6]
                #print(camera_name)
                cameras_gt_file = os.path.join(args.gt_path, f'{camera_name}.txt')

                cameras_gt = read_camera_file(cameras_gt_file)['poses']
                cameras_gt = cameras_gt.reshape(-1, 3, 4)

                results = calc_camera_error(images_bin, cameras_gt,
                                image_start_idx = args.image_start_idx)

                saved_results['sample'].append(images_bin)
                saved_results['camera'].append(camera_name)
                for k, v in results.items():
                    saved_results[k].append(v)
                 
            except Exception as e:
                print('Error', e)
                continue

    df = pd.DataFrame(saved_results)
    eval_dir = image_path #os.path.join('../../', colmap_folder)
    df.to_csv(os.path.join(eval_dir, 'eval.csv'), float_format='%.4f', index=False)
    print('save to', os.path.join(eval_dir, 'eval.csv'))
    print('rot_error_mean', df['rot_error_mean'].mean())
    metric_names = df.columns.tolist()
    for name in ['sample', 'camera']:
        metric_names.remove(name)

    stats = df.groupby('camera').agg({name: ['mean'] for name in metric_names})
    stats = stats.sort_values(by=('rot_error_mean','mean'), ascending=True)    
    stats.to_csv(os.path.join(eval_dir, 'eval_stats.csv'), float_format='%.4f', index=True)



