#!/usr/bin/env python
# coding: utf-8

# In[110]:


import os
import numpy as np
from abc import abstractmethod
import json
import warnings
import random
from PIL import Image

from flash_s3_dataloader.s3_io import (
    _get_s3_client,
    load_s3_image,
    load_s3_text,
    load_s3_json,
    check_s3_exists,
    list_s3_dir,
)

import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T


import sys
notebook_dir = os.path.dirname(os.path.abspath('__file__'))
parent_dir = os.path.dirname(notebook_dir)
sys.path.append(parent_dir)
from opensora.datasets.utils import center_crop_arr
from opensora.datasets import video_transforms
from opensora.datasets.camera_utils import invertRT

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


# In[50]:


class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    Args:
            n_attempts (int): number of sampling attempts before quite random sampling.
    """

    def __init__(self, n_attempts=3):      
        self.n_attempts = n_attempts

    def __call__(self, n_samples, n_frames):    
        sample_success = False
        attempt_count = 0
        while attempt_count < self.n_attempts and not sample_success: 
            rand_end = max(0, n_frames - n_samples - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + n_samples, n_frames)
            sample_success = end_index - begin_index == n_samples 
            attempt_count += 1
        if not sample_success:
            begin_index = 0
            end_index = n_samples - 1
        return begin_index, end_index

def get_transforms_video(resolution=512):
    transform = T.Compose(
        [
            video_transforms.ToTensorVideo(),  # TCHW
            #video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(resolution),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    return transform

class Loader(Dataset):
    def __init__(self, shuffle=False):
        super().__init__()
        self.shuffle = shuffle
    
    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind, msg='', verbose=False):
        if verbose:
            warnings.warn(f"Skipping index {ind}. {msg}")
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    @abstractmethod
    def __getitem__(self, ind):
        pass

class DatasetFromImages(Loader):
    """
    RealEstate10k dataset
    """

    def __init__(
        self,
        version='v0.5',
        phase="train",
        resolution=512, # only square for now
        sample_rate: int=2,
        caption_dir='',
        #caption_dir='/sensei-fs/users/scheong/datasets/RealEstate10k/captions',
        random_start_frame=False,
        n_samples=16,
        temporal_sampler=TemporalRandomCrop(),
        dropout=0.,
    ):
        super().__init__()
        self.random_start_frame = random_start_frame
        self.n_samples = n_samples

        assert version in ['v0.5', # 1280x720
                           'v0.7'] # 640x360
        
        file_lists = {}
        file_lists['v0.5'] = {
            'train': 's3://phidias/kaiz/dataset_paths/realestate10k_train_v0.5.txt',
            'test': 's3://phidias/kaiz/dataset_paths/realestate10k_test_v0.5.txt'
        }
        file_lists['v0.7'] = {
            'train': 's3://phidias/kaiz/dataset_paths/realestate10k_v0.7_train.txt',
            'test': 's3://phidias/kaiz/dataset_paths/realestate10k_v0.7_test_full.txt'
        }        
        self.caption_s3_path = 's3://phidias/chunhao/RealEstate10k_caption/cameractrl/'
        
        self.s3_client = _get_s3_client()
        self.dataset_dir=f's3://phidias/kaiz/RealEstate10k_preprocessed/{version}'
        self.phase = phase
        self.s3_postfix = "opencv_cameras.json"
        self.dir_list = self.load_array_from_txt(file_lists[version][phase], self.s3_postfix)
        self.sample_rate = sample_rate
        self.temporal_sampler = temporal_sampler
        self.resolution = resolution
        self.video_transform = get_transforms_video(resolution)
        self.captions = self._load_caption(caption_dir if caption_dir else self.caption_s3_path)
        self.dropout = dropout

    def _load_caption(self, caption_dir):
        
        caption_fname = f'{self.phase}_captions.json'
        try:
            caption_fname = os.path.join(caption_dir, caption_fname)
            if '/sensei-fs' in caption_dir:
                with open(caption_fname) as f:
                    captions = json.load(f)
            elif 's3://' in caption_dir:
                captions = load_s3_json(caption_fname)
            
            print(f'Loaded caption from {caption_fname}')
            missing = len(list(set(captions.keys()).symmetric_difference(set(self.dir_list))))
            total = len(list(self.dir_list))
            missed_rate = 100*missing/total

            warnings.warn(f"{missing}/{total} ({missed_rate:.1f}%) missing captions.")
        except Exception as e:
            warnings.warn(f"No caption detected at {caption_dir}", e)
            captions = None
        return captions

    def load_array_from_txt(self, txt_path, postfix=""):
        if '/sensei-fs' in txt_path:
            with open(txt_path) as f:
                lines = f.readlines()
        elif 's3://' in txt_path:
            lines = load_s3_text(txt_path, self.s3_client).strip().split('\n')

        lines = [
                os.path.basename(os.path.normpath(x.strip()[: x.strip().rfind(postfix)]))
                for x in lines
            ]

        return lines

    def __len__(self):
        return len(self.dir_list)

    def _get_caption(self, idx):
        # TO FIX, currently skip if has no caption 
        dir_name = self.dir_list[idx]  
        prompt = ['']
        if self.captions:
            prompt = self.captions.get(dir_name, [''])
            # if prompt is None:
            #     return self.skip_sample(idx)
        return prompt[0]
        
    def __getitem__(self, idx):

        try:

            prompt = self._get_caption(idx)
            dir_name = self.dir_list[idx]          
            s3_folder = os.path.join(
                self.dataset_dir,
                self.phase,
                dir_name
            )

            metadata  = load_s3_json(os.path.join(s3_folder, self.s3_postfix))

            if metadata['frames'] == None:
                self.skip_sample(idx, 'None frames', verbose=False)
            frames = metadata['frames']

            H, W = frames[0]['h'], frames[0]['w']
            if min(H, W) < self.resolution:
                return self.skip_sample(idx, f"Video resolution {H}x{W} lower than {self.resolution}", verbose=True)

            ''' Temporal sampling '''
            if self.sample_rate > 1:
                frames = frames[::self.sample_rate]
            start_frame_ind, end_frame_ind  = self.temporal_sampler(self.n_samples, len(frames))
            if end_frame_ind - start_frame_ind < self.n_samples:
                return self.skip_sample(idx, f'Insufficient number of frames. {end_frame_ind - start_frame_ind}')
            frames = frames[start_frame_ind: end_frame_ind]

            ''' Load images '''
            image_filenames = [os.path.join(s3_folder, frame['file_path']) for frame in frames]
            imgs = np.array([load_s3_image(img_path, self.s3_client) for img_path in image_filenames])
            T, H, W, C = imgs.shape
            if min(H, W) < self.resolution:
                return self.skip_sample(idx, f"Physical Video resolution {H}x{W} lower than {self.resolution}", verbose=True)
            video = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2)

            ''' Image transformation '''
            video = self.video_transform(video)
            # TCHW -> CTHW
            #video = video.permute(1, 0, 2, 3)
            # for visualize output
            video = video.permute(0, 2, 3, 1) * 0.5 + 0.5

            ''' Load rotation and translation matrices '''
            camera = np.array([frame['w2c'] for frame in frames], dtype=np.float32)
            #camera = invertRT(camera)
            #camera = camera[:,:-1,:].flatten(-2,-1)

            if random.random() < self.dropout:
                camera = torch.from_numpy(np.tile(np.eye(4, dtype=np.float32)[:-1].flatten(), (16,1)))
                video = video[:,0,:,:].unsqueeze(1).repeat(1, 16, 1, 1)

            assert prompt != None
            assert video != None
            return {'video': video, 
                    'text': prompt,
                    'camera':camera,
                    'images':imgs,
                    'frames':frames,
                    'dir_name':dir_name
                    }
        except Exception as e:
            self.skip_sample(idx, msg=e, verbose=True)


ds = DatasetFromImages('v0.7', phase='train', resolution=256, sample_rate=2)


def save_data(data, folder):
    def format_line(f):
        w, h = f['w'], f['h']
        fx = f['fx']/w
        fy = f['fy']/h
        cx = f['cx']/w
        cy = f['cy']/h
        t = 0
        numbers = [fx, fy, cx, cy, 0., 0.] + list(np.array(f['w2c'])[:-1].flatten())

        formatted_numbers = [str(t)]+[f"{num:.9f}" for num in numbers]
        line = ' '.join(formatted_numbers)
        return line

    # write camera file
    camera_pose = [format_line(f) for f in data['frames']]
    dir_name = data['dir_name']
    subdir = os.path.join(folder, dir_name)
    os.makedirs(subdir, exist_ok=True)

    text_file = os.path.join(subdir, 'camera.txt')

    with open(text_file, 'w') as file:
        file.write(dir_name+'\n')
        for pose in camera_pose:
            file.write(pose + '\n')
    
    # create gif
    gif_path = os.path.join(subdir, 'video.gif')
    video_numpy = (data['video'] * 255).cpu().numpy().astype(np.uint8) 
    images = [Image.fromarray(v) for v in video_numpy]
    images[0].save(gif_path, save_all=True, append_images=images[1:], loop=0, duration=120)


for i in tqdm(range(50)):
    save_data(ds[i], '/sensei-fs/users/scheong/camera_viz')


