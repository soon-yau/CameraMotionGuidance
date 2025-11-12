#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-machine inference with local-only distributed init (world_size=1).

Key points:
- Uses file:// rendezvous (no DNS, no sockets), so hostnames like ig217.igreat.com are irrelevant.
- Forces rank=0, world_size=1.
- Sets safe NCCL/GLOO env (loopback, no IB) to avoid network issues on any host.
"""

import os
import pickle
import tempfile
import atexit
from glob import glob
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
from mmengine.runner import set_random_seed

from opensora.datasets import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import to_torch_dtype
from opensora.datasets.camera_utils import (
    read_camera_file,
    get_relative_pose,
    add_homogeneous,
    adjust_intrinsic,
    compute_ray_cond,
    scale_max_dist_to_1,
)

# --------- Local-only, robust distributed init (rank=0, world=1) ----------
def init_distributed_local_file_store():
    """
    Initialize torch.distributed with a file:// init_method so that
    no hostname/IP resolution or sockets are needed. Works on any machine.
    """
    if not dist.is_available() or dist.is_initialized():
        return

    # Force single-process values so any downstream checks succeed.
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    # Make NCCL/GLOO stick to loopback and avoid IB – safe anywhere.
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    # Use a temp file for rendezvous so there’s zero networking.
    tmpdir = tempfile.mkdtemp(prefix="torch_store_")
    store_path = os.path.join(tmpdir, "shared_init")
    init_method = f"file://{store_path}"

    # Clean up the temp dir at exit
    @atexit.register
    def _cleanup_tmpdir():
        try:
            # Best-effort cleanup
            for fn in os.listdir(tmpdir):
                try:
                    os.remove(os.path.join(tmpdir, fn))
                except Exception:
                    pass
            os.rmdir(tmpdir)
        except Exception:
            pass

    print(f"[dist] initializing (local file store): backend={backend}, init_method={init_method}")
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=0,
        world_size=1,
    )


def finalize_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
        print("[dist] destroyed process group")


def load_prompts(prompt_path):
    with open(prompt_path, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_cameras_pickle(camera_path):
    camera = []
    try:
        with open(camera_path, 'rb') as f:
            camera.append(pickle.load(f))
    except Exception:
        pass
    return camera


def main():
    # ---- initialize local-only dist so ColossalAI/ShardFormer is satisfied
    init_distributed_local_file_store()

    # ======================================================
    # 1) cfg and basic runtime
    # ======================================================
    cfg = parse_configs(training=False)

    # Override train script values for inference
    if getattr(cfg, "disable_cache", False):
        if isinstance(getattr(cfg, "vae", None), dict):
            cfg.vae["cache_dir"] = None
        if isinstance(getattr(cfg, "text_encoder", None), dict):
            cfg.text_encoder["cache_dir"] = None

    # Disable CFG if dataset.static_camera_rate == 0
    disable_cfg = False
    if getattr(getattr(cfg, "dataset", None), "static_camera_rate", 1) == 0:
        if isinstance(cfg.scheduler, dict):
            cfg.scheduler["cfg_scale_c"] = 0
        disable_cfg = True
        print("WARNING: Classifier-Free Guidance DISABLED due to static_camera_rate == 0")

    # Fix sampling steps unless overridden
    if isinstance(cfg.scheduler, dict):
        cfg.scheduler["num_sampling_steps"] = 100

    print(cfg)

    # ======================================================
    # 2) runtime: device/dtype/seed/prompts
    # ======================================================
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(getattr(cfg, "dtype", "fp16"))
    set_random_seed(seed=getattr(cfg, "seed", 42))

    prompts = load_prompts(cfg.prompt_path)

    # ======================================================
    # 3) build model & load weights
    # ======================================================
    input_size = (cfg.num_frames, *cfg.image_size)

    vae = build_module(cfg.vae, MODELS)
    latent_size = vae.get_latent_size(input_size)

    # Text encoder often wants fp32 internally; device is respected inside
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)

    # No sequence parallelism in single-process mode
    enable_sequence_parallelism = False

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
    # hack for classifier-free guidance
    text_encoder.y_embedder = model.y_embedder

    vae = vae.to(device, dtype).eval()
    model = model.to(device, dtype).eval()

    camera_encoder = None
    if 'camera_encoder' in cfg:
        camera_encoder = build_module(cfg.camera_encoder, MODELS, from_pretrained=cfg.camera_ckpt_path)
        camera_encoder = camera_encoder.to(device, dtype).eval()

    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # Multi-resolution support
    model_args = {}
    if getattr(cfg, "multi_resolution", False):
        image_size = cfg.image_size
        hw = torch.tensor([image_size], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        ar = torch.tensor([[image_size[0] / image_size[1]]], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        model_args["data_info"] = dict(ar=ar, hw=hw)

    # ======================================================
    # 4) inference
    # ======================================================
    nprompts = min(len(prompts), cfg.nprompts)

    # Collect camera files
    if os.path.isfile(cfg.camera_path):
        camera_files = [cfg.camera_path]
    else:
        camera_files = glob(os.path.join(cfg.camera_path, '**/*.txt'), recursive=True)
        camera_files.sort(key=len)

    cameras = [read_camera_file(f) for f in camera_files]
    print('cameras', camera_files)

    for camera in cameras:
        subdir = os.path.basename(camera['filename']).split('.')[0]
        save_dir_root = os.path.join(cfg.save_dir, subdir)

        camera_pose = camera['poses']
        intrinsics = camera['intrinsics'][0]

        # Pose preprocessing
        camera_pose = camera_pose.reshape(-1, 3, 4)
        camera_pose = add_homogeneous(camera_pose)        # (F, 4, 4)
        camera_pose = get_relative_pose(camera_pose)      # (F, 4, 4)
        camera_pose = torch.from_numpy(camera_pose)       # to torch
        camera_pose = scale_max_dist_to_1(camera_pose)    # normalize translation scale

        if getattr(cfg.dataset, "plucker_coord", False):
            h, w = cfg.image_size
            fx, fy = intrinsics['fx'] * w, intrinsics['fy'] * h
            cx, cy = intrinsics['cx'] * w, intrinsics['cy'] * h
            fx, fy, cx, cy = adjust_intrinsic((h, w), h, fx, fy, cx, cy)
            fxfycxcy = torch.FloatTensor([fx, fy, cx, cy]).expand(camera_pose.shape[0], -1)
            camera_pose = compute_ray_cond(h=h, w=w, fxfycxcy=fxfycxcy, c2w=camera_pose)   # (F, 6, H, W)
        else:
            camera_pose = camera_pose[:, :-1, :]  # (F, 3, 4)
            camera_pose = camera_pose.reshape(-1, 12)

        # Optional sparse control (off)
        drop_flag = False
        if not drop_flag:
            prefix = ''
            drop_sets = [[]]
        else:
            prefix = 'drop_'
            drop_sets = [
                [],
                [1, 3, 5, 7, 9, 11, 13] + [2, 6, 10, 14] + [4, 12],
                [8],
            ]
        drop_frame_indices = []

        camera_pose_copy = deepcopy(camera_pose)
        for drop_set in drop_sets:
            set_random_seed(seed=getattr(cfg, "seed", 42))
            drop_frame_indices += drop_set

            if drop_flag:
                save_dir = os.path.join(save_dir_root, f'{prefix}{len(drop_frame_indices)}')
            else:
                save_dir = save_dir_root

            camera_pose = camera_pose_copy.clone()
            for idx in drop_frame_indices:
                camera_pose[idx].zero_()

            camera_pose = camera_pose.unsqueeze(0)  # (1, F, 12) or (1, F, 6, H, W)

            if getattr(cfg.dataset, "expand_rt", False):
                assert not getattr(cfg.dataset, "plucker_coord", False), \
                    "expand_rt should be False when using plucker_coord"
                camera_pose = (
                    camera_pose
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .repeat(1, 1, 1, *cfg.image_size)
                )

            if camera_encoder is not None:
                camera_pose_encoded = camera_encoder(camera_pose.to(device, dtype))
                if getattr(cfg.dataset, "plucker_coord", False):
                    h, w = cfg.image_size
                    fx, fy = intrinsics['fx'] * w, intrinsics['fy'] * h
                    cx, cy = intrinsics['cx'] * w, intrinsics['cy'] * h
                    fx, fy, cx, cy = adjust_intrinsic((h, w), h, fx, fy, cx, cy)
                    fxfycxcy = torch.FloatTensor([fx, fy, cx, cy]).expand(camera_pose.shape[1], -1)
                    c2w = torch.eye(4, 4).expand(camera_pose.shape[1], -1, -1)
                    camera_null = compute_ray_cond(h=h, w=w, fxfycxcy=fxfycxcy, c2w=c2w)
                    camera_null = camera_encoder(camera_null.unsqueeze(0).to(device, dtype))
                else:
                    camera_null = torch.eye(3, 4).view(1, 1, -1, 1, 1).repeat(
                        camera_pose_encoded.shape[0],
                        camera_pose_encoded.shape[1],
                        1, 1, 1
                    ).to(device, dtype)
                    camera_null = camera_encoder(camera_null)
                camera_pose = camera_pose_encoded
            else:
                if getattr(cfg.dataset, "plucker_coord", False):
                    h, w = cfg.image_size
                    fx, fy = intrinsics['fx'] * w, intrinsics['fy'] * h
                    cx, cy = intrinsics['cx'] * w, intrinsics['cy'] * h
                    fx, fy, cx, cy = adjust_intrinsic((h, w), h, fx, fy, cx, cy)
                    fxfycxcy = torch.FloatTensor([fx, fy, cx, cy]).expand(camera_pose.shape[1], -1)
                    c2w = torch.eye(4, 4).expand(camera_pose.shape[1], -1, -1)
                    camera_null = compute_ray_cond(h=h, w=w, fxfycxcy=fxfycxcy, c2w=c2w)   # (F, 6, H, W)
                    camera_null = camera_null.unsqueeze(0).to(device, dtype)
                else:
                    camera_null = torch.eye(3, 4).view(1, 1, -1).expand(camera_pose.shape).to(device, dtype)

            if disable_cfg:
                camera_null = camera_pose

            os.makedirs(save_dir, exist_ok=True)
            sample_idx = 0
            SAMPLE_NAME_OFFSET = 0

            for i in range(0, nprompts, cfg.batch_size):
                batch_prompts = prompts[i: i + cfg.batch_size]

                samples = scheduler.sample(
                    model,
                    text_encoder,
                    z_size=(vae.out_channels, *latent_size),
                    prompts=batch_prompts,
                    device=device,
                    additional_args=model_args,
                    cameras=camera_pose.to(device, dtype),
                    camera_null=camera_null.to(device, dtype),
                    positive_prompt=getattr(cfg, "pos_prompt", None),
                    negative_prompt=None,
                )

                samples = vae.decode(samples.to(dtype))

                for idx, sample in enumerate(samples):
                    print(f"Prompt: {batch_prompts[idx]}")
                    sampledir = f"sample_{sample_idx + SAMPLE_NAME_OFFSET}"
                    save_path = os.path.join(save_dir, sampledir)
                    save_image_path = os.path.join(save_dir, f'images/{sampledir}')
                    save_sample(
                        sample,
                        fps=cfg.fps,
                        save_path=save_path,
                        save_image_path=save_image_path
                    )
                    sample_idx += 1

    finalize_distributed()


if __name__ == "__main__":
    main()
