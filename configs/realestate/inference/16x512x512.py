num_frames = 16
fps = 30 // 3
image_size = (512, 512)

# Define model
CACHE_DIR=None
PRETRAINED_MODEL = '/home/scheong/sensei-fs-link/github/Open-Sora/ckpts/epoch9-global_step34000/ema.pt'
model = dict(
    type="STDiT-XL/2",
    space_scale=1.0,
    time_scale=1.0,
    enable_flashattn=True,
    enable_layernorm_kernel=True,
    enable_camera_control=False,
    from_pretrained=PRETRAINED_MODEL,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
    micro_batch_size=128,
    cache_dir=CACHE_DIR
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=120,
    shardformer=True,
    cache_dir=CACHE_DIR,
)
scheduler = dict(
    type="iddpm",
    #num_sampling_steps=100,
    cfg_scale=7.0,
    cfg_channel=3, # or None
)
dtype = "fp16"

# Others
batch_size = 1
seed = 42
prompt_path = "./assets/texts/realestate10k.txt"
save_dir = "./outputs/samples/"

camera_path = ''
nprompts = None
#camera_encoder = None
save_dir = None