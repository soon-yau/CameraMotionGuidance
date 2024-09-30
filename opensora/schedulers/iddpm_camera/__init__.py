from functools import partial

import torch

from opensora.registry import SCHEDULERS

from ..iddpm import IDDPM

@SCHEDULERS.register_module("iddpm_camera")
class IDDPM_CAMERA(IDDPM):
    def __init__(
        self,
        cfg_scale_t,
        cfg_scale_c,
        *args,
        **kwargs,
    ):
        self.cfg_scale_t = cfg_scale_t
        self.cfg_scale_c = cfg_scale_c
        super().__init__(*args, **kwargs)

    def sample(
        self,
        model,
        text_encoder,
        z_size,
        prompts,
        device,
        additional_args=None,
        cameras=None,
        camera_null=None,
        positive_prompt='',
        negative_prompt=None

    ):
        n = len(prompts)
        z = torch.randn(n, *z_size, device=device)
        z = torch.cat([z, z, z], 0)

        # text
        prompts = [ positive_prompt + prompt for prompt in prompts]
        model_args = text_encoder.encode(prompts)
        y_text = model_args["y"]

        if not negative_prompt:
            y_null = text_encoder.null(n)
        else:
            y_null = text_encoder.encode(n*[negative_prompt])['y']

        # Camera
        expand_dims = [n] + [-1] * (len(cameras.shape) - 1)
        camera_pose = cameras.expand(*expand_dims).to(device)
        camera_null = camera_null.expand(*expand_dims).to(device)
        
        cond_text = []
        cond_camera = []
        # (Null, Null)
        cond_text.append(y_null)
        cond_camera.append(camera_null)

        # (text, null)
        cond_text.append(y_text)
        cond_camera.append(camera_null)

        # (text, camera)
        cond_text.append(y_text)
        cond_camera.append(camera_pose)
            
        model_args["y"] = torch.cat(cond_text, 0)
        model_args['camera'] = torch.cat(cond_camera, 0)
         
        if additional_args is not None:
            model_args.update(additional_args)

        forward = partial(forward_with_cfg, model, 
                          cfg_scale_t=self.cfg_scale_t,
                          cfg_scale_c=self.cfg_scale_c,
                          cfg_channel=self.cfg_channel)
        samples = self.p_sample_loop(
            forward,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_args,
            progress=True,
            device=device,
        )
        samples, _, _ = samples.chunk(3, dim=0)
        return samples


def forward_with_cfg(model, x, timestep, y, cfg_scale_t, cfg_scale_c, cfg_channel=None, **kwargs):
    # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
    z = x[: len(x) // 3]
    combined = torch.cat([z, z, z], dim=0)
    
    model_out = model.forward(combined, timestep, y, **kwargs)
    model_out = model_out["x"] if isinstance(model_out, dict) else model_out

    if cfg_channel is None:
        cfg_channel = model_out.shape[1] // 2
    eps, rest = model_out[:, :cfg_channel], model_out[:, cfg_channel:]
    uncond_eps, cond_eps_t, cond_eps_c = torch.split(eps, len(eps) // 3, dim=0)
    partial_eps = uncond_eps + cfg_scale_t * (cond_eps_t - uncond_eps) + \
                               cfg_scale_c * (cond_eps_c - cond_eps_t)

    eps = torch.cat([partial_eps, partial_eps, partial_eps], dim=0)
    return torch.cat([eps, rest], dim=1)

