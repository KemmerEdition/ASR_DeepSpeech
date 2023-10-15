from hw_asr.augmentations.base import AugmentationBase
import torchaudio
from torch import Tensor
import numpy as np


class TimeStretch(AugmentationBase):
    def __init__(self, p, fix_rate, *args, **kwargs):
        self.p = p
        self.fix_rate = fix_rate
        self.aug_time_str = torchaudio.transforms.TimeStretch(*args, **kwargs)

    def __call__(self, data: Tensor):
        p_ = np.random.binomial(1, self.p)
        x = data.unsqueeze(1)
        if p_ > 0.5:
            return data
        else:
            return self.aug_time_str(x, self.fix_rate).squeeze(1)
