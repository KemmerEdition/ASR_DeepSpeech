from hw_asr.augmentations.base import AugmentationBase
import torchaudio
from torch import Tensor
import numpy as np


class PitchShift(AugmentationBase):
    def __init__(self, p, sample_rate, n_steps, **kwargs):
        self.p = p
        self.aug_pitch = torchaudio.transforms.PitchShift(sample_rate, n_steps)

    def __call__(self, data: Tensor):
        p_ = np.random.binomial(1, self.p)
        x = data.unsqueeze(1)
        if p_:
            return self.aug_pitch(x).squeeze(1)
        else:
            return data
