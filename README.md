
## Boosting Camera Motion Control for Video Diffusion Transformers

## Install
```
sudo apt-get install git-lfs libgl1 -y
conda create -n cmg python=3.10
conda activate cmg
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/soon-yau/apex

pip install -v .

```

## Inference
1. Download model folder e.g. /cameratrlcmg from https://huggingface.co/soonyau/cmg/tree/main to "./models"
2. Run the script and pass in the model path e.g. `bash scripts/inference.sh models/cameractrlcmg`. Change camera pose and text prompt in config.py in model path.

## Citation

```bibtex
        @article{cheong2024cmg,
        author    = {Cheong, Soon Yau and Mustafa, Armin and Ceylan, Duygu and Gilbert, Andrew and Huang, Chun-hao Paul},
        title     = {Boosting Camera Motion Control for Video Diffusion Transformers},
        journal   = {Arxiv Preprint 2410.10802},
        month     = {October},
        year      = {2024}}
```
