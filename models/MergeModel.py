import sys
sys.path.append(".")
import torch
from torch import nn
from models.TimesNet import Model as TimeNet
from models.XiNet6_splitMK import Model as XiNet6_splitMK
from models.DLinear import Model as DLinear

class Model(nn.Module):    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.timesnet = TimeNet(configs)
        self.xinet = XiNet6_splitMK(configs)
        self.dlinear = DLinear(configs)
        
        self.fc = nn.Sequential(
            nn.Linear(configs.num_class * 2, configs.num_class)
        )
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out = []
        # out.append(
        #     self.timesnet(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        # )
        out.append(
            self.xinet(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        )
        out.append(
            self.dlinear(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        )
        out = torch.cat(out, dim=1)
        out = self.fc(out)
        return out