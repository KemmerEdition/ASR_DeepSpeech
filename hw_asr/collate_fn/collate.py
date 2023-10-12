import logging
from typing import List
from collections import defaultdict
import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch = defaultdict(list)
    # TODO: your code here
    for i in dataset_items:
        result_batch['text_encoded_length'].append(i['text_encoded'].shape[1])
        result_batch['spectrogram_length'].append(i['spectrogram'].shape[2])

        result_batch['spectrogram'].append(i['spectrogram'].squeeze(0).transpose(0, -1))
        result_batch['text_encoded'].append(i['text_encoded'].squeeze(0).transpose(0, -1))
        result_batch['text'].append(i['text'])

        result_batch['audio'] += [i['audio'].squeeze(0)]
        result_batch['duration'].append(i['duration'])
        result_batch['audio_path'].append(i['audio_path'])

    for v in result_batch:
        if v in ['text_encoded_length', 'spectrogram_length']:
            result_batch[v] = torch.Tensor(result_batch[v]).long()
        elif v == 'spectrogram':
            result_batch[v] = pad_sequence(result_batch[v], batch_first=True, padding_value=0.0).transpose(1, -1)
        elif v == 'text_encoded':
            result_batch[v] = pad_sequence(result_batch[v], batch_first=True, padding_value=0.0).transpose(1, -1)
        elif (len(result_batch[v])) > 0 and (v == 'audio'):
            result_batch[v] = pad_sequence(result_batch[v], batch_first=True, padding_value=0.0).transpose(1, -1)

    return result_batch
