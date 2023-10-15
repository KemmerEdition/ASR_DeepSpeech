from hw_asr.augmentations.base import AugmentationBase
import torch_audiomentations
from torch import Tensor
import numpy as np


class PitchShift(AugmentationBase):
    def __init__(self, p, *args, **kwargs):
        assert 0 <= p <= 1
        self.p = p
        self.aug_pitch = torch_audiomentations.PitchShift(*args, **kwargs)

    def __call__(self, data: Tensor):
        p_ = np.random.binomial(1, self.p)
        x = data.unsqueeze(1)
        if p_:
            return self.aug_pitch(x).squeeze(1)
        else:
            return data
