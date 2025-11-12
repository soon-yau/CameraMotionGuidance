
## Boosting Camera Motion Control for Video Diffusion Transformers

# Installtion
## System
```
pip install uv
sudo apt update && sudo apt install -y git git-lfs
git lfs install
```

## Environment
```
uv python install 3.10
uv sync

# (Optional) Add acceleration extras. Do not currently work now.
# uv sync --extra flash --extra xformers --extra apex

source .venv/bin/activate 

```

After installation, we suggest reading [structure.md](docs/structure.md) to learn the project structure and how to use the config files.


# Inference
# Download models
1. Download model folder e.g. /cameratrlcmg from https://huggingface.co/soonyau/cmg/tree/main to "./models"
```
MODEL_NAME="cameractrlcmg"
huggingface-cli download soonyau/cmg --repo-type=model --local-dir ./models/${MODEL_NAME}$ --include "${MODEL_NAME}$/*"
```
Go to config.py and disable all acceleration optimization
```
    enable_flashattn=False,
    enable_layernorm_kernel=False,
    
    shardformer=False,

```

2. Run the script and pass in the model path e.g. `bash scripts/inference.sh models/cameractrlcmg`. Change camera pose and text prompt in config.py in model path.

## Citation
```bibtex
        @inproceedings{cheong2024cmg,
        author    = {Cheong, Soon Yau and Mustafa, Armin and Ceylan, Duygu and Gilbert, Andrew and Huang, Chun-hao Paul},
        title     = {Boosting Camera Motion Control for Video Diffusion Transformers},
        booktitle = {Proceedings of the British Machine Vision Conference (BMVC)},
        year      = {2025}}
```
