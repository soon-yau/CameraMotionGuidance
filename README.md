
## Boosting Camera Motion Control for Video Diffusion Transformers

## Install
```
git clone https://github.com/soon-yau/apex.git
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

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
