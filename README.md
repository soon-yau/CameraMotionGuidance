
# Boosting Camera Motion Control for Video Diffusion Transformers
Accepted into BMVC 2025.
Project Page: https://soon-yau.github.io/CameraMotionGuidance/

Existing camera control methods for U-Net do not work for diffusion transformers. We developed the first camera control for space-time diffuser transformer. Our method, based on classifier-free guidance, restore controllability and boost motion by over 400%.

### 🚀 Camera Motion Guidance
 Conventionally, no guidance is used in Classifier-free guidance (CFG) or camera condition is supplied without a motion reference.  In training, we randomly set some video be static (repeat first frame to whole sequence) and all-zeros camera condition, which we use as null motion reference. Then we use a seperate camera guidance term to allow for control independent to text conditioning.

$$
\begin{aligned}
\hat{e_{\theta}}(z_t, C_T, C_C) &= e_{\theta}(z_t, \emptyset_T, \emptyset_C) \\
&\quad + s_T \{ e_{\theta}(z_t, C_T, \emptyset_C) - e_{\theta}(z_t, \emptyset_T, \emptyset_C) \} \\
&\quad + s_C \{ e_{\theta}(z_t, C_T, C_C) - e_{\theta}(z_t, C_T, \emptyset_C) \}
\end{aligned}
$$


### 🎥 Conditioning Camera Poses

<p>Conditioning camera poses.</p>

<p align="center">
  <img src="supplementary/1_Method_Comparison_files/000.png" alt="Pose 000" width="18%">
  <img src="supplementary/1_Method_Comparison_files/001.png" alt="Pose 001" width="18%">
  <img src="supplementary/1_Method_Comparison_files/002.png" alt="Pose 002" width="18%">
  <img src="supplementary/1_Method_Comparison_files/003.png" alt="Pose 003" width="18%">
  <img src="supplementary/1_Method_Comparison_files/004.png" alt="Pose 004" width="18%">
</p>

---

### ⚠️ MotionCtrl in DiT — *Uncontrollable Motion*

<p>MotionCtrl method in DiT has uncontrollable motion.</p>

<p align="center">
  <img src="supplementary/1_Method_Comparison_files/050.gif" alt="MotionCtrl sample 050" width="18%">
  <img src="supplementary/1_Method_Comparison_files/053.gif" alt="MotionCtrl sample 053" width="18%">
  <img src="supplementary/1_Method_Comparison_files/056.gif" alt="MotionCtrl sample 056" width="18%">
  <img src="supplementary/1_Method_Comparison_files/059.gif" alt="MotionCtrl sample 059" width="18%">
  <img src="supplementary/1_Method_Comparison_files/062.gif" alt="MotionCtrl sample 062" width="18%">
</p>

---

### 🚧 CameraCtrl in DiT — *Limited Motion*

<p>CameraCtrl method in DiT has limited motion.</p>

<p align="center">
  <img src="supplementary/1_Method_Comparison_files/051.gif" alt="CameraCtrl sample 051" width="18%">
  <img src="supplementary/1_Method_Comparison_files/054.gif" alt="CameraCtrl sample 054" width="18%">
  <img src="supplementary/1_Method_Comparison_files/057.gif" alt="CameraCtrl sample 057" width="18%">
  <img src="supplementary/1_Method_Comparison_files/060.gif" alt="CameraCtrl sample 060" width="18%">
  <img src="supplementary/1_Method_Comparison_files/063.gif" alt="CameraCtrl sample 063" width="18%">
</p>

---

### 🚀 Our Method — *Restored Controllability and Boosted Motion*

<p>Our method restores camera controllability with boosted motion.</p>

<p align="center">
  <img src="supplementary/1_Method_Comparison_files/052.gif" alt="Ours sample 052" width="18%">
  <img src="supplementary/1_Method_Comparison_files/055.gif" alt="Ours sample 055" width="18%">
  <img src="supplementary/1_Method_Comparison_files/043.gif" alt="Ours sample 043" width="18%">
  <img src="supplementary/1_Method_Comparison_files/061.gif" alt="Ours sample 061" width="18%">
  <img src="supplementary/1_Method_Comparison_files/064.gif" alt="Ours sample 064" width="18%">
</p>




## Citation
```bibtex
        @inproceedings{cheong2024cmg,
        author    = {Cheong, Soon Yau and Mustafa, Armin and Ceylan, Duygu and Gilbert, Andrew and Huang, Chun-hao Paul},
        title     = {Boosting Camera Motion Control for Video Diffusion Transformers},
        booktitle = {Proceedings of the British Machine Vision Conference (BMVC)},
        year      = {2025}}
```

## Installation
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


