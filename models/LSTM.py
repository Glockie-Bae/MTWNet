import sys
sys.path.append("..")
import torch.nn as nn
from layers.Embed import DataEmbedding

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.emb = DataEmbedding(configs.enc_in, configs.d_model)
        
        self.lstm = nn.LSTM(configs.d_model, configs.d_model, batch_first=True)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(configs.d_model*configs.seq_len, configs.num_class),
        )
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x = self.emb(x_enc, None)
        h0, _ = self.lstm(x)
        out = self.fc(h0)
        return out