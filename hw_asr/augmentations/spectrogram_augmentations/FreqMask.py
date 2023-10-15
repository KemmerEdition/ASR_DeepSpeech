from hw_asr.augmentations.base import AugmentationBase
import torchaudio
from torch import Tensor
import numpy as np


class FreqMask(AugmentationBase):
    def __init__(self, p, freq_mask_param, *args, **kwargs):
        self.p = p
        self.aug_freq = torchaudio.transforms.FrequencyMasking(freq_mask_param)

    def __call__(self, data: Tensor):
        p_ = np.random.binomial(1, self.p)
        x = data.unsqueeze(1)
        if p_ > 0.5:
            return data
        else:
            return self.aug_freq(x).squeeze(1)
