from torch import nn

from hw_asr.base import BaseModel

class Model_For_Overfit(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, **batch):
        super().__init__(n_feats, n_class, **batch)

        self.gru = nn.GRU(n_feats, fc_hidden, num_layers=3,  bidirectional=True)
        self.fc_1 = nn.Linear(fc_hidden * 2, n_class)

    def forward(self, spectrogram, **batch):
        x, _ = self.gru(spectrogram)
        x = self.fc_1(x)
        return {"logits": x}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here