import os
import pickle
import torch
from glob import glob
import colossalai
import torch.distributed as dist
from mmengine.runner import set_random_seed
import numpy as np
from copy import deepcopy

from opensora.datasets import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import to_torch_dtype
from opensora.datasets.camera_utils import read_camera_file, get_relative_pose, add_homogeneous, adjust_intrinsic, compute_ray_cond, scale_max_dist_to_1
from opensora.acceleration.parallel_states import set_sequence_parallel_group
from colossalai.cluster import DistCoordinator

def load_prompts(prompt_path):
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts

def load_cameras_pickle(camera_path):
    camera = []
    try:
        with open(camera_path, 'rb') as f:
            camera.append(pickle.load(f))
    except:
        pass
    return camera
    

# Use load
def main():
    # ======================================================
    # 1. cfg and init distributed env
    # ======================================================
    cfg = parse_configs(training=False)

    # Override train script for inference
    if cfg.disable_cache:
        cfg.vae['cache_dir'] = None
        cfg.text_encoder['cache_dir'] = None

    # scheme with no cfg
    disable_cfg = False
    if cfg.dataset.static_camera_rate == 0:
        cfg.scheduler['cfg_scale_c'] = 0
        disable_cfg = True
        print("WARNING!!!!!!! CFG DISABLED!")

    cfg.scheduler['num_sampling_steps'] = 100
    print(cfg)
    camera_format = cfg['CAMERA_FORMAT']
    # init distributed
    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()

    if coordinator.world_size > 1:
        set_sequence_parallel_group(dist.group.WORLD) 
        enable_sequence_parallelism = True
    else:
        enable_sequence_parallelism = False

    # ======================================================
    # 2. runtime variables
    # ======================================================
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.dtype)
    set_random_seed(seed=cfg.seed)
    prompts = load_prompts(cfg.prompt_path)

    # ======================================================
    # 3. build model & load weights
    # ======================================================
    # 3.1. build model
    input_size = (cfg.num_frames, *cfg.image_size)
    vae = build_module(cfg.vae, MODELS)
    latent_size = vae.get_latent_size(input_size)
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)  # T5 must be fp32

    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        dtype=dtype,
        enable_sequence_parallelism=enable_sequence_parallelism,
    )
    text_encoder.y_embedder = model.y_embedder  # hack for classifier-free guidance

    # 3.2. move to device & eval
    vae = vae.to(device, dtype).eval()
    model = model.to(device, dtype).eval()

    camera_encoder = None
    if 'camera_encoder' in cfg:
        camera_encoder = build_module(cfg.camera_encoder, MODELS, from_pretrained=cfg.camera_ckpt_path)
        camera_encoder = camera_encoder.to(device, dtype).eval()

    # 3.3. build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # 3.4. support for multi-resolution
    model_args = dict()
    if cfg.multi_resolution:
        image_size = cfg.image_size
        hw = torch.tensor([image_size], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        ar = torch.tensor([[image_size[0] / image_size[1]]], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        model_args["data_info"] = dict(ar=ar, hw=hw)

    # ======================================================
    # 4. inference
    # ======================================================
    
    nprompts = min(len(prompts), cfg.nprompts)
    if os.path.isfile(cfg.camera_path):
        camera_files = [cfg.camera_path]
    else:
        camera_files = glob(os.path.join(cfg.camera_path, '**/*.txt'), recursive=True)
        camera_files.sort(key=len)

    cameras = [read_camera_file(f) for f in camera_files]
    print('cameras', camera_files)

    for camera in cameras:

        #for t_scale in [1]:
        subdir = os.path.basename(camera['filename']).split('.')[0]

        #subsubdir = str(t_scale).replace('.','_')
        #subdir = os.path.join(f't_scale_{subsubdir}', subdir)
        save_dir_root = os.path.join(cfg.save_dir, subdir)
        camera_pose = camera['poses']
        intrinsics = camera['intrinsics'][0]

        # do something here
        camera_pose = camera_pose.reshape(-1, 3, 4)
        camera_pose = add_homogeneous(camera_pose)
        camera_pose = get_relative_pose(camera_pose)
        camera_pose = torch.from_numpy(camera_pose)
        camera_pose = scale_max_dist_to_1(camera_pose)

        if cfg.dataset.plucker_coord:
            ## assuming all frames have the same intrinsics, adjust them here and copy-paste them to each frame later
            
            h, w = cfg.image_size
            fx, fy = intrinsics['fx'] * w, intrinsics['fy'] * h
            cx, cy = intrinsics['cx'] * w, intrinsics['cy'] * h
            fx, fy, cx, cy = adjust_intrinsic((h, w), h, fx, fy, cx, cy)
            fxfycxcy = torch.FloatTensor([fx, fy, cx, cy]).expand(camera_pose.shape[0], -1)
            #c2w = [torch.inverse(camera[i]) for i in range(camera.shape[0])]
            #c2w = torch.stack(c2w, dim=0)           # F, 4, 4
            camera_pose = compute_ray_cond(h = h, w = w, fxfycxcy=fxfycxcy, c2w=camera_pose)   # F, 6, H, W
        else:
            camera_pose = camera_pose[:,:-1,:]
            camera_pose = camera_pose.reshape(-1, 12)

        # drop frames for sparse control
        drop_flag = False
        if not drop_flag:            
            prefix = ''
            drop_sets = [[]]
        else:
            prefix = 'drop_'
            drop_sets = [\
                [],
                [1,3,5,7,9,11,13]+
                [2,6,10,14] + \
                [4, 12],
                [8]
            ]            
        drop_frame_indices = []

        camera_pose_copy = deepcopy(camera_pose)
        for drop_set in drop_sets:
            set_random_seed(seed=cfg.seed)
            drop_frame_indices += drop_set

            if drop_flag:
                save_dir = os.path.join(save_dir_root, f'{prefix}{len(drop_frame_indices)}')
            else:
                save_dir = save_dir_root
            camera_pose = camera_pose_copy
            #drop_frame_indices = [1,
            #                       3,5,7,9,11,13]+[2,4,6,8,10,12,14]
            for idx in drop_frame_indices:
                camera_pose[idx].zero_()

            camera_pose = camera_pose.unsqueeze(0)
            if cfg.dataset.expand_rt:
                assert cfg.dataset.plucker_coord==False
                camera_pose = camera_pose.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,*cfg.image_size)

            camera_pose_shape = list(camera_pose.shape)
            if camera_encoder is not None:                        
                camera_pose = camera_encoder(camera_pose.to(device, dtype))
                if cfg.dataset.plucker_coord:
                    c2w = torch.eye(4, 4).expand(camera_pose.shape[1], -1, -1)
                    camera_null = compute_ray_cond(h = h, w = w, fxfycxcy=fxfycxcy, c2w=c2w)   # F, 6, H, W
                    camera_null = camera_encoder(camera_null.unsqueeze(0).to(device, dtype))
                else:                
                    camera_pose_shape[2] = 1
                    camera_null = torch.eye(3,4).view(1,1,-1,1,1).repeat(camera_pose_shape).to(device, dtype)
                    camera_null = camera_encoder(camera_null)
            #if camera_format == 'extrinsic':
            else:
                camera_null = torch.eye(3,4).view(1,1,-1).expand(camera_pose.shape).to(device, dtype)

            if disable_cfg:
                camera_null = camera_pose            

            try:
                os.makedirs(save_dir, exist_ok=False)
            except:
                pass
            sample_idx = 0

            SAMPLE_NAME_OFFSET = 0

            for i in range(0, nprompts, cfg.batch_size):
                batch_prompts = prompts[i : i + cfg.batch_size]
                samples = scheduler.sample(
                    model,
                    text_encoder,
                    z_size=(vae.out_channels, *latent_size),
                    prompts=batch_prompts,
                    device=device,
                    additional_args=model_args,
                    cameras=camera_pose,
                    camera_null=camera_null,
                    positive_prompt=cfg.pos_prompt,
                    negative_prompt=None
                )
                samples = vae.decode(samples.to(dtype))

                if coordinator.is_master():
                    for idx, sample in enumerate(samples):
                        print(f"Prompt: {batch_prompts[idx]}")
                        sampledir = f"sample_{sample_idx+SAMPLE_NAME_OFFSET}"
                        save_path = os.path.join(save_dir, sampledir)
                        save_image_path = os.path.join(save_dir, f'images/{sampledir}')
                        save_sample(sample, fps=cfg.fps, 
                                    save_path=save_path, 
                                    save_image_path=save_image_path)
                        sample_idx += 1


if __name__ == "__main__":
    main()