## User specific 
proj_name = 'dit-cam-v03-plucker-2d' # wandb project
outputs = "/sensei-fs/users/scheong/outputs/"  # save checkpoint   
wandb_api_key = 'local-26bd83ee4d3349052c5147a5e867e2659b66c6a6'
CACHE_DIR='/sensei-fs/users/scheong/.cache/huggingface/hub'
#CACHE_DIR=None
PRETRAINED_MODEL='/sensei-fs/users/scheong/github/Open-Sora/ckpts/OOO/OpenSora-v1-16x256x256.pth'
## end of user specific 

num_frames = 16
frame_interval = 1
image_size = (256, 256)
fps = 30//2  # for inference
plucker_scale = image_size[0]//16

# Define dataset
root = None
data_path = "CSV_PATH"
use_image_transform = False
num_workers = 6
# Define acceleration
dtype = "fp16"
grad_checkpoint = False
plugin = "zero2"
sp_size = 1
data_prefetch = 1

MODEL_DIM = 1152
CAMERA_FORMAT = 'plucker'

# Define model
model = dict(
    type="STDiT-XL/2",
    space_scale=0.5,
    time_scale=1.0,
    from_pretrained=PRETRAINED_MODEL,
    enable_flashattn=True,
    enable_layernorm_kernel=True,
    camera_fuser_linear_dims=[MODEL_DIM, MODEL_DIM],
    camera_format=CAMERA_FORMAT
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
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
    cfg_scale_t=6.0,
    cfg_scale_c=4.0
)

camera_encoder = dict(
    type="PluckerEncoder",
    downscale_factor=plucker_scale,
    channels=[MODEL_DIM],
    nums_rb=2,
    cin=12 * plucker_scale**2, 
    ksize=1,
    sk=True,
    use_conv=False,
    compression_factor=1,
    temporal_attention_nhead=8,
    attention_block_types=["Temporal_Self", ],
    temporal_position_encoding=True,
    temporal_position_encoding_max_len=16    
)

dataset = dict(
    text_dropout=0.05,
    camera_dropout=0.05,
    static_camera_rate=0.05,
    resolution=256,
    version='v0.7',
    frame_strides=[4, 5, 6, 7, 8],
    plucker_coord=False,
    expand_rt=True
    )

# Others
seed = 42
wandb = True

epochs = 8
log_every = 300
ckpt_every = 2000

load = None
batch_size = 5 # 80GB - 6
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