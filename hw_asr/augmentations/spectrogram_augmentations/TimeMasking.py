from hw_asr.augmentations.base import AugmentationBase
import torchaudio
from torch import Tensor
import numpy as np


class TimeMasking(AugmentationBase):
    def __init__(self, p, time_mask_param, *args, **kwargs):
        self.p = p
        self.aug_time_mask = torchaudio.transforms.TimeMasking(time_mask_param)

    def __call__(self, data: Tensor):
        p_ = np.random.binomial(1, self.p)
        x = data.unsqueeze(1)
        if p_ > 0.5:
            return data
        else:
            return self.aug_time_mask(x).squeeze(1)