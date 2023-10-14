from torch import nn
from torch.nn import Sequential
from hw_asr.base import BaseModel


# class BatchRNN(nn.Module):
#     def __int__(self, input_size, hidden_size):
#         super(BatchRNN).__int__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.batch_norm = nn.BatchNorm1d(hidden_size)
#         self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, bidirectional=True)
#
#     def forward(self, x, h=None):
#         x, h = self.rnn(x, h)
#         if self.bidirectional:
#             x = x.view(x.size(0), x.size(1), 2, -1).sum(2)
#         t, n = x.size(0), x.size(1)
#         x = x.view(t * n, -1)
#         x = self.module(x)
#         x = x.view(t, n, -1)
#         return x, h
# Делала упрощенную версию того, что представляет архитектура по ссылке ниже
# https://github.com/SeanNaren/deepspeech.pytorch/blob/master/deepspeech_pytorch/model.py

class DeepSpeech2(BaseModel):
    def __init__(self, n_feats, n_class, hidden, num_rnn, **batch):
        super().__init__(n_feats, n_class, **batch)

        self.convs = Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 96, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )

        rnn_input_size = (n_feats - 41) // 2 + 1
        rnn_input_size = (rnn_input_size - 21) // 2 + 1
        rnn_input_size = (rnn_input_size - 21) // 2 + 1
        rnn_input_size *= 96

        self.batch_norm = nn.BatchNorm1d(hidden * 2)
        self.rnn = nn.RNN(input_size=rnn_input_size, hidden_size=hidden, num_layers=num_rnn, bidirectional=True)
        # self.lookahead = Sequential(
        #     nn.Conv1d(self.hidden, self.hidden), nn.ReLU()
        # )
        self.lin = nn.Linear(in_features=hidden * 2, out_features=n_class)

    def forward(self, spectrogram, **batch):

        x = self.convs(spectrogram.transpose(1, 2).unsqueeze(1))
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = x.transpose(1, 2)
        x, y = self.rnn(x)
        x = x.transpose(1, 2)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)
        x = self.lin(x)

        return {"logits": x}

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2