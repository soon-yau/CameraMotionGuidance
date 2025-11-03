
## Boosting Camera Motion Control for Video Diffusion Transformers

## Install
```
git clone https://github.com/soon-yau/apex.git
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

## Installation

```bash
# create a virtual env
conda create -n cmg python=3.10
conda activate cmg

# install torch
# the command below is for CUDA 12.1, choose install commands from 
# https://pytorch.org/get-started/locally/ based on your own CUDA version
#pip3 install torch torchvision
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# install flash attention (optional)
pip install packaging ninja
pip install flash-attn --no-build-isolation

# install apex (optional)
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git

# install xformers
pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121
#pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121

# install this project
git clone https://github.com/hpcaitech/Open-Sora
cd Open-Sora
pip install -v .
```

After installation, we suggest reading [structure.md](docs/structure.md) to learn the project structure and how to use the config files.


## Inference
1. Download model folder e.g. /cameratrlcmg from https://huggingface.co/soonyau/cmg/tree/main to "./models"
2. Run the script and pass in the model path e.g. `bash scripts/inference.sh models/cameractrlcmg`. Change camera pose and text prompt in config.py in model path.

## Citation

```bibtex
@software{opensora,
  author = {Soon Yau Cheong and Chun-Hao Paul Huang and Duygu Ceylan and Armin Mustafa and Andew Gilbert},
  title = {Boosting Camera Motion Control for Video Diffusion Transformers},
  month = {October},
  year = {2024},
}
```
