from typing import List
import random
import numpy as np
import torch
import torchvision.transforms.v2 as T
from . import video_transforms


def scale_max_dist_to_1(camera: torch.Tensor): # camera: F, 4, 4
    scalar = camera[:,:3,3].norm(dim=1).max()
    if scalar > 1e-1:
        camera[:,:3,3] /= scalar
    return camera

def compute_ray_cond(h, w, fxfycxcy, c2w):
    # h, w: height, width of input image [before tiling i.e., 512]
    # fxfycxcy: [v, 4]
    # c2w: [v, 4, 4]
    # assert h == 512 and w == 512, '@yiwhu: hard-coded for 2x2 grid generation!'
    
    v = fxfycxcy.shape[0]
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    
    # [h, w]
    y_norm, x_norm = (y + 0.5) / h * 2 - 1, (x + 0.5) / w * 2 - 1
    # [b, v, 2, h, w]
    xy_norm = torch.stack([x_norm, y_norm], dim=0)[None, :, :, :].expand(
        v, -1, -1, -1
    )

    x = x[None, :, :].expand(v, -1, -1).reshape(v, -1)
    y = y[None, :, :].expand(v, -1, -1).reshape(v, -1)
    x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    ray_d = torch.stack([x, y, z], dim=2)  # [v, h*w, 3]
    ray_d_cam = ray_d.clone()
    ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [v, h*w, 3]
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [v, h*w, 3]
    ray_d_cam = ray_d_cam / torch.norm(ray_d_cam, dim=2, keepdim=True)
    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [v, h*w, 3]

    ray_o = ray_o.reshape(v, h, w, 3).permute(0, 3, 1, 2) # [v, 3, h, w]
    ray_d = ray_d.reshape(v, h, w, 3).permute(0, 3, 1, 2) # [v, 3, h, w]
    
    ray_condition = torch.cat(
        [
            torch.cross(ray_o, ray_d, dim=1),
            ray_d,
        ],
        dim=1,
    ) # [v, 6, h, w]
    
    return ray_condition

def adjust_intrinsic(orig_size, target_size, fx, fy, cx, cy):
    if isinstance(target_size, tuple):
        if len(target_size) != 2:
            raise ValueError(f"size should be tuple (height, width), instead got {size}")
        size = target_size
    else:
        size = (target_size, target_size)

    ## scaling
    H, W = orig_size[0], orig_size[1]
    scale_ = size[0] / min(H, W)

    ## cropping
    h, w = round(scale_ * H), round(scale_ * W)
    th, tw = size
    if h < th or w < tw:
        raise ValueError("height and width must be no smaller than crop_size")
    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))

    ## fx, fy, cx, cy
    return scale_ * fx, scale_ * fy, scale_ * cx - j, scale_ * cy - i

def add_homogeneous(camera_poses: np.array):
    n_dim = len(camera_poses.shape) 
    row = np.array([[0, 0, 0, 1]], dtype=camera_poses.dtype)
    assert n_dim == 2 or n_dim == 3
    if n_dim == 2:
        return np.concatenate((camera_poses, row), axis=0)
    elif n_dim == 3:
        new_camera_poses = []
        for pose in camera_poses:
            new_camera_poses.append(np.concatenate((pose, row), axis=0))
        return np.array(new_camera_poses)
    
def get_relative_pose(w2cs, zero_t_first_frame=True):
    source_cam_c2w = w2cs[0]
    if zero_t_first_frame:
        cam_to_origin = 0
    else:
        cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ np.linalg.inv(w2c) for w2c in w2cs[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses   

def get_inverted_pose(w2cs):
    inv = np.linalg.inv(w2cs[0])
    ret_poses = [inv @ w2c for w2c in w2cs]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses   


def invertRT(camera):
    camera = torch.from_numpy(camera)
    det = torch.det(camera[0])
    if det == 0:
        raise ValueError("This camera matrix is not invertible")
    
    inverse_matrix = torch.inverse(camera[0])
    return torch.matmul(inverse_matrix, camera)

def read_camera_file(camera_file):
    with open(camera_file, 'r') as f:
        lines = f.readlines()
    
    intrinsics = []
    poses = []
    for line in lines[1:]:
        line = np.array(line.replace('\n','').split(' '), dtype=np.float32)
        fx, fy, cx, cy = line[1:5]
        intrinsics.append({'fx':fx, 'fy':fy, 'cx':cx, 'cy':cy})
        poses.append(line[7:])
    return {'intrinsics':intrinsics, 
            'poses':np.array(poses),
            'filename': camera_file}

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

class RandomReverseVideo:
    def __init__(self, probability:float=0.5):
        self.probability = probability
        
    def __call__(self, frames: List):
        if random.random() < self.probability:
            return frames[::-1]
        else:
            return frames

class RandomTemporalSample:
    def __init__(self, 
                 min_frames: int,
                 frame_strides:List = [1], 
                 frame_strides_ratios:List=[]):
        if len(frame_strides_ratios) == 0:
            frame_strides_ratios = [1] * len(frame_strides)

        self.min_frames = min_frames
        self.frame_strides = frame_strides
        self.frame_strides_ratios = frame_strides_ratios

    def _sample(self, frames):
        n_frames = len(frames)

        
        if n_frames <= self.min_frames:
            return frames

        sample_success = False
        attempt_count = 0
        while attempt_count < 3 and not sample_success: 
            rand_end = max(0, n_frames - self.min_frames - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.min_frames, n_frames)
            sample_success = end_index - begin_index == self.min_frames
            attempt_count += 1
        if not sample_success:
            begin_index = 0
            end_index = self.min_frames - 1

        return frames[begin_index:end_index]
        
    def __call__(self, frames):
        frame_len = len(frames)
        sample_weights = []
        for rate, ratio in zip(self.frame_strides, self.frame_strides_ratios):
            weight = ratio if frame_len // rate >= self.min_frames else 0
            sample_weights.append(weight)

        skip_rate = 1
        if sum(sample_weights) > 0.:
            skip_rate = random.choices(self.frame_strides, sample_weights)[0]

        return self._sample(frames[::skip_rate])
