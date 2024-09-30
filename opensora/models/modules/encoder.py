import torch
import torch.nn as nn
from opensora.registry import MODELS
from opensora.utils.ckpt_utils import load_checkpoint

@MODELS.register_module()
class CameraLinear(nn.Module):
    """
        Zero Initialized Linear.
        """
    def __init__(
            self,
            in_channel,
            out_channel
        ):
        super().__init__()
        self.proj = nn.Linear(in_channel, out_channel)
        # self.proj.weight.data.fill_(0)  # Initialize weights with zeros
        # self.proj.bias.data.fill_(0)   

    def forward(self, x):
        return self.proj(x)

@MODELS.register_module("CameraEncoder_1")
def CameraEncoder_1(from_pretrained=None, **kwargs):
    model = CameraLinear(**kwargs)
    if from_pretrained is not None:
        load_checkpoint(model, from_pretrained)
    return model
