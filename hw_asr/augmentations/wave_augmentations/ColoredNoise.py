import torch_audiomentations
from torch import Tensor
import numpy as np
from hw_asr.augmentations.base import AugmentationBase


class ColoredNoise(AugmentationBase):
    def __int__(self, p, *args, **kwargs):
        self.p = p
        self.aug_col = torch_audiomentations.AddColoredNoise(*args, **kwargs)

    def __call__(self, data: Tensor):
        p_ = np.random.binomial(1, self.p)
        x = data.unsqueeze(1)
        if p_:
            return self.aug_col(x).squeeze(1)
        else:
            return data
