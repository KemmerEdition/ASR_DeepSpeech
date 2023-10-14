from typing import List

import torch
import numpy as np
from torch import Tensor

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_cer



class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = self.text_encoder.decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)

class BeamSearchMetricCER(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, beam_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []
        lengths = log_probs_length.detach().numpy()
        predictions = np.exp(log_probs.detach().cpu().numpy())
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_beam_search"):
                pred_text = self.text_encoder.ctc_beam_search(log_prob_vec[:length], self.beam_size)
            else:
                pred_text = self.text_encoder.decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
#
# class BeamSearchLMMetricCER(BaseMetric):
#     def __init__(self, text_encoder: CTCCharTextEncoder, beam_size: int = 10, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.text_encoder = text_encoder
#         self.beam_size = beam_size
#
#     def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], beam_size: int, **kwargs):
#         cers = []
#         lengths = log_probs_length.detach().numpy()
#         for log_prob, length, target_text in zip(log_probs, lengths, text):
#             target_text = BaseTextEncoder.normalize_text(target_text)
#             hypos = self.text_encoder.ctc_beam_search_from_liba(log_prob.detach().cpu().numpy(), length, beam_size)
#             pred_text = hypos[0][0]
#             cers.append(calc_cer(target_text, pred_text))
#         return sum(cers) / len(cers)