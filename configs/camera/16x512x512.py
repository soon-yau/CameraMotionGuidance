## User specific 
proj_name = 'dit-cam-v03-camera-ext'
outputs = "/sensei-fs/users/scheong/outputs/"
wandb_api_key = 'local-26bd83ee4d3349052c5147a5e867e2659b66c6a6'
CACHE_DIR='/sensei-fs/users/scheong/.cache/huggingface/hub'
#CACHE_DIR=None
PRETRAINED_MODEL='/sensei-fs/users/scheong/github/Open-Sora/ckpts/OOO/OpenSora-v1-HQ-16x512x512.pth'
## end of user specific 

num_frames = 16
frame_interval = 1
image_size = (512, 512)
fps = 30//2  # for inference

# Define dataset
root = None
data_path = "CSV_PATH"
use_image_transform = False
num_workers = 2

# Define acceleration
dtype = "fp16"
grad_checkpoint = False
plugin = "zero2"
sp_size = 1
data_prefetch = 2


MODEL_DIM = 1152
CAMERA_FORMAT = 'extrinsic'
CAMERA_PARAMS_NUM = 12

# Define model
model = dict(
    type="STDiT-XL/2",
    space_scale=1.0,
    time_scale=1.0,
    from_pretrained=PRETRAINED_MODEL,
    enable_flashattn=True,
    enable_layernorm_kernel=True,
    camera_fuser_linear_dims=[MODEL_DIM+CAMERA_PARAMS_NUM, MODEL_DIM],
    camera_format=CAMERA_FORMAT
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
    micro_batch_size=128,
    cache_dir=CACHE_DIR,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=120,
    shardformer=True,
    cache_dir=CACHE_DIR,
)
scheduler = dict(
    type="iddpm_camera",
    #num_sampling_steps=100,
    cfg_scale_t=4.0,
    cfg_scale_c=15.0
)

# Others
seed = 42
wandb = True

epochs = 15
log_every = 200
ckpt_every = 2000

dataset = dict(
    text_dropout=0.05,
    camera_dropout=0.05,
    static_camera_rate=0.05,
    resolution=256,
    version='v0.7',
    frame_strides=[4, 5, 6, 7, 8],
    plucker_coord=False
)

load = None
batch_size = 1
lr = 1e-5 
grad_clip = 1.0
freeze_model = True
active_layer_names = ['camera_fuser', 'attn_temp']

# Inference
prompt_path = "./assets/texts/realestate10k.txt"
#prompt_path = "./assets/texts/t2v_sora.txt"

camera_path = ''
nprompts = None
save_dir = None