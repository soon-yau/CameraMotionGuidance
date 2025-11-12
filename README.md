
# Boosting Camera Motion Control for Video Diffusion Transformers

Project Page: https://soon-yau.github.io/CameraMotionGuidance/
## Citation
```bibtex
        @inproceedings{cheong2024cmg,
        author    = {Cheong, Soon Yau and Mustafa, Armin and Ceylan, Duygu and Gilbert, Andrew and Huang, Chun-hao Paul},
        title     = {Boosting Camera Motion Control for Video Diffusion Transformers},
        booktitle = {Proceedings of the British Machine Vision Conference (BMVC)},
        year      = {2025}}
```

## Installtion
```
pip install uv
uv python install 3.10
uv sync
source .venv/bin/activate 
```

## Download models
Download model folder e.g. /cameratrlcmg from https://huggingface.co/soonyau/cmg/tree/main to "./models"
```
MODEL_NAME="cameractrlcmg"
huggingface-cli download soonyau/cmg --repo-type=model --local-dir ./models/${MODEL_NAME}$ --include "${MODEL_NAME}$/*"
```

Disable acceleration extras in config.py as compiling GPU-specific CUDA extensions can fail with incompatible build environment. 
```
    enable_flashattn=False,
    enable_layernorm_kernel=False,
    ...
    shardformer=False,
```

## Inference
Run the script and pass in the model path e.g. `bash scripts/inference.sh models/cameractrlcmg`. Change camera pose and text prompt in config.py in model path.


