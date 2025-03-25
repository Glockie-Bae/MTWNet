import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models.NLinear as DLinear
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
from Modules.Down_wt import Down_wt
from scipy.signal import savgol_filter
from models.FAN import  FAN


class Model(nn.Module):

    # FITS: Frequency Interpolation Time Series Forecasting

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = 0
        self.individual = 0
        self.channels = configs.enc_in

        self.freq_len = int(self.seq_len // 2 + 1)
        self.dominance_freq = self.freq_len // 4  # 720/24
        self.length_ratio = (self.seq_len + self.pred_len)/self.seq_len

        self.down_wt = Down_wt(
            configs.enc_in, configs.enc_in)

        self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(
            torch.cfloat)

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer_norm = nn.LayerNorm(configs.d_model)


        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(nn.Linear(self.dominance_freq, int(self.dominance_freq*self.length_ratio)).to(torch.cfloat))

        else:
            self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq*self.length_ratio)).to(torch.cfloat) # complex layer for frequency upcampling]
        # configs.pred_len=configs.seq_len+configs.pred_len
        # #self.Dlinear=DLinear.Model(configs)
        # configs.pred_len=self.pred_len
        self.my_projection = nn.Linear(
            configs.enc_in * 2 * (self.seq_len // 2), configs.num_class)

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=8, stride=4),
            nn.BatchNorm1d(256)
        )
        self.downsample_layers.append(stem)

        self.conv1 = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

        self.conv2 = nn.Sequential(
            Inception_Block_V1(configs.enc_in, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.enc_in,
                               num_kernels=configs.num_kernels)
        )

        self.conv1d = torch.nn.Conv1d(in_channels=22, out_channels=6, kernel_size=1)

        self.nor_FAN = FAN(self.seq_len, self.pred_len, configs.enc_in, freq_topk = 3)

        self.ircom = nn.Linear(configs.enc_in * 2, configs.enc_in)





    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B,L,C = x.shape

        # RIN
        # 使其均值为0
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
        # print(x_var)
        x = x / torch.sqrt(x_var)


        low_specx = torch.fft.rfft(x, dim=1)


        low_specx[:,self.dominance_freq:]=0 # LPF
        low_specx = low_specx[:,0:self.dominance_freq,:] # LPF


        # print(low_specx.permute(0,2,1))
        if self.individual:
            low_specxy_ = torch.zeros([low_specx.size(0),int(self.dominance_freq*self.length_ratio),low_specx.size(2)],dtype=low_specx.dtype).to(low_specx.device)
            for i in range(self.channels):
                low_specxy_[:,:,i]=self.freq_upsampler[i](low_specx[:,:,i].permute(0,1)).permute(0,1)
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0,2,1)).permute(0,2,1)
        # print(low_specxy_)
        P_B, P_L, P_C = low_specxy_.size(0), int((self.seq_len + self.pred_len) / 2 + 1), low_specxy_.size(2)

        low_specxy = torch.zeros([P_B, P_L,P_C],dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:,0:low_specxy_.size(1),:]=low_specxy_ # zero padding


        low_xy = torch.fft.irfft(low_specxy, dim=1)


        low_xy=low_xy * self.length_ratio # energy compemsation for the length change
        #dom_x=x-low_x
        
        #dom_xy=self.Dlinear(dom_x)
        #xy=(low_xy+dom_xy) * torch.sqrt(x_var) +x_mean # REVERSE RIN

        # low_xy = low_xy.permute(0, 2, 1).unsqueeze(2)
        # low_xy = self.conv(low_xy).squeeze(2).permute(0,2,1)


        #xy = low_xy
        xy=(low_xy) * torch.sqrt(x_var) +x_mean
        output = xy.reshape(xy.shape[0], -1)
        output = self.my_projection(output)  # (batch_size, num_classes)



        return output

