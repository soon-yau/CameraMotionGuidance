num_frames = 16
frame_interval = 1
image_size = (512, 512)

# Define dataset
root = None
data_path = "CSV_PATH"
use_image_transform = False
num_workers = 8

# Define acceleration
dtype = "bf16"
grad_checkpoint = False
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="STDiT-XL/2",
    space_scale=1.0,
    time_scale=1.0,
    from_pretrained=None,
    enable_flashattn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
    micro_batch_size=128,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=120,
    shardformer=True,
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)

# Others
seed = 42
outputs = "/sensei-fs/users/scheong/outputs"
wandb = True
wandb_api_key = 'local-26bd83ee4d3349052c5147a5e867e2659b66c6a6'
wandb_proj = 'OpenSoraRealEstate'

epochs = 20
log_every = 100
ckpt_every = 2000
load = None

batch_size = 2
lr = 2e-5
grad_clip = 1.0
