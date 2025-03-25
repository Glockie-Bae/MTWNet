import sys
sys.path.append("..")
from typing import Optional, Union
from torch import nn
from tsai.models import ROCKET_Pytorch


# 不确定代码是否正确，ROCKET_Pytorch文档太少了
class Model(nn.Module):
    def __init__(self, configs, n_kernels=10000):
        super(Model, self).__init__()
        self.device = configs.device
        self.n_kernels = n_kernels
        self.rocket = ROCKET_Pytorch.ROCKET(c_in=configs.enc_in, seq_len=configs.seq_len, n_kernels=self.n_kernels)
        self.fc = nn.Sequential(
            # nn.Dropout(configs.dropout),
            nn.BatchNorm1d(20000),
            nn.Linear(20000, configs.num_class)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x_feat = self.rocket(x_enc.transpose(1, 2)).to(x_enc.device)
        out = self.fc(x_feat)
        return out
    