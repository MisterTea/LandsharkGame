from typing import Dict, List, Tuple
from uuid import UUID, uuid4

import numpy as np
import pytorch_lightning as pl
import torch


class FullyConnectedForward(torch.nn.Module):
    def __init__(self, backModule):
        super().__init__()
        self.layers = torch.nn.ModuleList([x.cpu() for x in backModule.layers])
        self.activations = backModule.activations
        # self.forward = backModule.forward
        self.forward_cache_dict = {}

    def forward(self, inputs):
        x = inputs
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if self.activations[i] is not None:
                x = self.activations[i](x)
        return x

    def forward_cache(self, inputs):
        input_key = hashlib.blake2b(inputs.numpy().tobytes()).digest()
        labels = self.forward_cache_dict.get(input_key, None)
        if labels is None:
            x = self.forward(inputs)
            self.forward_cache_dict[input_key] = x
            return x
        else:
            return labels
