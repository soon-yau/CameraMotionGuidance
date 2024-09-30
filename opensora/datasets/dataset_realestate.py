import os
import numpy as np
from abc import abstractmethod
import json
import warnings
import random
from einops import rearrange

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
from torchvision.transforms import transforms

from .camera_utils import get_relative_pose, get_transforms_video, adjust_intrinsic, compute_ray_cond, RandomReverseVideo, RandomTemporalSample

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
        caption_dir='',
        #caption_dir='/sensei-fs/users/scheong/datasets/RealEstate10k/captions',
        video_len=16,
        text_dropout=0.,
        camera_dropout=0.,
        static_camera_rate=0.05,
        frame_strides=[4, 8],
        frame_strides_ratios=[],
        plucker_coord=False,
        expand_rt=False,
    ):
        super().__init__()
        self.video_len = video_len

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
        self.temporal_transform = transforms.Compose([
                        RandomTemporalSample(min_frames=video_len, 
                                             frame_strides=frame_strides,
                                             frame_strides_ratios=frame_strides_ratios),
                        RandomReverseVideo(probability=0.5)
                        ])
        self.resolution = resolution
        self.video_transform = get_transforms_video(resolution)
        self.captions = self._load_caption(caption_dir if caption_dir else self.caption_s3_path)
        self.text_dropout = text_dropout
        self.camera_dropout = camera_dropout
        self.static_camera_rate = static_camera_rate
        self.plucker_coord = plucker_coord
        self.expand_rt = expand_rt
        
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
                self.skip_sample(idx, 'None frames')
            frames = metadata['frames']

            H, W = frames[0]['h'], frames[0]['w']
            if min(H, W) < self.resolution:
                return self.skip_sample(idx, f"Video resolution {H}x{W} lower than {self.resolution}")

            ''' Temporal sampling '''
            frames = self.temporal_transform(frames)
            if len(frames) < self.video_len:
                return self.skip_sample(idx, f'Insufficient number of frames. {len(frames)}')

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
            video = video.permute(1, 0, 2, 3)

            ''' Load rotation and translation matrices '''
            camera = np.array([frame['w2c'] for frame in frames], dtype=np.float32)

            #camera = invertRT(camera)
            camera = get_relative_pose(camera)
            camera = torch.from_numpy(camera)

            if self.plucker_coord:
                ''' Adjust intrinsic according to the cropping & resizing'''
                ## assuming all frames have the same intrinsics, adjust them here and copy-paste them to each frame later
                fx, fy = frames[0]['fx'], frames[0]['fy']
                cx, cy = frames[0]['cx'], frames[0]['cy']
                h, w = frames[0]['h'], frames[0]['w']

                fx, fy, cx, cy = adjust_intrinsic((h, w), self.resolution, fx, fy, cx, cy)

                fxfycxcy = torch.FloatTensor([fx, fy, cx, cy]).expand(camera.shape[0], -1)
                #c2w = [torch.inverse(camera[i]) for i in range(camera.shape[0])]
                #c2w = torch.stack(c2w, dim=0)           # F, 4, 4
                camera = compute_ray_cond(h = self.resolution, w = self.resolution, fxfycxcy=fxfycxcy, c2w=camera)   # F, 6, H, W
                #camera = rearrange(camera, 'c f h w -> f c h w')
            else:
                camera = camera[:,:-1,:].flatten(-2,-1) # F, 12

            ''' Data Augmentation ''' 
            if random.random() < self.text_dropout:
                prompt = ''

            if random.random() < self.static_camera_rate:
                if self.plucker_coord:
                    c2w = torch.eye(4, 4).expand(camera.shape[0], -1, -1)
                    camera = compute_ray_cond(h = self.resolution, w = self.resolution, fxfycxcy=fxfycxcy, c2w=c2w)   # F, 6, H, W
                else:
                    camera = torch.from_numpy(np.tile(np.eye(3, 4, dtype=np.float32).flatten(), (self.video_len, 1)))
                video = video[:,0,:,:].unsqueeze(1).repeat(1, self.video_len, 1, 1)
            elif random.random() < self.camera_dropout:
                indices = list(range(1,camera.shape[0]-1))
                num_to_drop = random.randint(int(0.7*len(indices)), len(indices))
                frames_to_drop = random.sample(indices, num_to_drop)
                for i in frames_to_drop:
                    camera[i].zero_()

            if self.expand_rt:
                camera = camera.unsqueeze(-1).unsqueeze(-1).repeat(1,1,self.resolution,self.resolution)


            return {'video': video, 
                    'text': prompt,
                    'camera':camera,
                    #'frames':frames,
                    #'dir_name':dir_name                    
                    #'images':imgs
                    }
        except Exception as e:
            self.skip_sample(idx, msg=e, verbose=True)


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    dataset_config = dict(
        text_dropout=0.05,
        camera_dropout=0.5,
        static_camera_rate=0.05,
        resolution=512,
        version='v0.5',
        frame_strides=[4, 5, 6, 7, 8],
        plucker_coord=True,
        expand_rt=False
    )

    train_dataset = DatasetFromImages(**dataset_config)
    x = train_dataset[0]
    print(x['camera'].shape)
    import pdb; pdb.set_trace()

    train_dataloader = DataLoader(
                                train_dataset,
                                batch_size=1,
                                num_workers=1,
                                shuffle=False,
                                worker_init_fn=None,
                                pin_memory=True,
                            )


    dl = iter(train_dataloader)
    data = next(dl)
    breakpoint()  
    # for bi, batch in tqdm(enumerate(train_dataloader)):
    #     print(1)