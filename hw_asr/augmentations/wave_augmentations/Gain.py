import torch_audiomentations
from torch import Tensor
import numpy as np

from hw_asr.augmentations.base import AugmentationBase
# добавила вероятность в аргументы


class Gain(AugmentationBase):
    def __init__(self, p, *args, **kwargs):
        assert 0 <= p <= 1
        self.p = p
        self.augmentation = torch_audiomentations.Gain(*args, **kwargs)

    def __call__(self, data: Tensor):
        p_ = np.random.binomial(1, self.p)
        x = data.unsqueeze(1)
        if p_:
            return self.augmentation(x).squeeze(1)
        else:
            return data
