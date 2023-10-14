from torch import nn
import torch.nn.functional as F
from hw_asr.base import BaseModel

class DeepSpeechRNN(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, **batch):
        super().__init__(n_feats, n_class, **batch)

        self.fc_1 = nn.Linear(in_features=n_feats,
                              out_features=fc_hidden, bias=True)
        self.fc_2 = nn.Linear(in_features=fc_hidden,
                              out_features=fc_hidden, bias=True)
        self.fc_3 = nn.Linear(in_features=fc_hidden,
                              out_features=fc_hidden, bias=True)
        self.fc_4 = nn.Linear(in_features=fc_hidden,
                              out_features=fc_hidden, bias=True)
        self.dropout = nn.Dropout(p=0.1)
        self.hid = fc_hidden
        self.RNN = nn.RNN(input_size=fc_hidden,
                          hidden_size=fc_hidden,
                          bidirectional=True)
        self.fc_5 = nn.Linear(in_features=fc_hidden,
                              out_features=n_class, bias=True)

    def forward(self, spectrogram, **batch):
        x = self.fc_1(spectrogram.transpose(1, 2))
        x = F.hardtanh(F.relu(x), 0, 20)
        x = self.dropout(x)

        x = self.fc_2(x)
        x = F.hardtanh(F.relu(x), 0, 20)
        x = self.dropout(x)

        x = self.fc_3(x)
        x = F.hardtanh(F.relu(x), 0, 20)
        x = x.squeeze(0).transpose(0, 1)

        x, y = self.RNN(x)
        x = x[:, :, :self.hid] + x[:, :, self.hid:]

        x = self.fc_4(x)
        x = F.hardtanh(F.relu(x), 0, 20)
        x = self.dropout(x)

        x = self.fc_5(x).permute(1, 0, 2)

        return x

    def transform_input_lengths(self, input_lengths):
        return input_lengths

