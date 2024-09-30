import torch.nn as nn
from cameractrl.models import CameraPoseEncoder
from opensora.registry import MODELS
from opensora.utils.ckpt_utils import load_checkpoint

@MODELS.register_module('PluckerEncoder')
def PluckerEncoder(from_pretrained=None, **kwargs):
    model = CameraPoseEncoder(**kwargs)
    if from_pretrained is not None:
        load_checkpoint(model, from_pretrained)
    return model
