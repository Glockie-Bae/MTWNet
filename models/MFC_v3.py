import math
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F

from Modules.Down_wt_v3 import Down_wt
from layers.Conv_Blocks import Inception_Block_V1

def calLen(f1, f2):
    if f1 % 2 == 0:
        f1 = f1 / 2
    else:
        f1 = (f1 // 2) + 1
    if f2 % 2 == 0:
        f2 = f2 / 2
    else:
        f2 = (f2 // 2) + 1
    return f1 * f2

class Model(nn.Module):


    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = 1
        self.channels = configs.enc_in
        self.task_name = configs.task_name
        self.downsample_Layers = 2
        self.freq_len = int(self.seq_len // 2 + 1)
        self.dominance_freq = self.freq_len // 4   # 720/24
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )
        self.down_wt = Down_wt(
            configs.enc_in, configs.enc_in)

        #self.wt_upsampler = nn.Linear(self.freq_len, self.seq_len)  # complex layer for frequency upcampling]

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(
                    nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(torch.cfloat))

        else:
            self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)).to(
                torch.cfloat)  # complex layer for frequency upcampling]


        self.wt_channel = 0
        len = self.seq_len
        for i in range(self.downsample_Layers + 1):
            f1, f2 = self.find_closest_factors(len)
            self.wt_channel += calLen(f1, f2)
            len = len // 2

        self.WTLinear = nn.Linear(int(self.wt_channel), int((self.seq_len + self.pred_len) / 2 + 1))
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.enc_in * 2 * (self.seq_len // 2), configs.num_class)

    def MulitProcessInput(self, x):
        x = x.permute(0,2,1)
        x_list = []
        x_list.append(x)
        for i in range(self.downsample_Layers):
            downsampled_data = F.avg_pool1d(x, kernel_size=2, stride=2)
            x_list.append(downsampled_data)
            x = downsampled_data
        return x_list

    def MulitConvWT(self, x_list):

        for i in range(len(x_list)):
            x_list[i] = self.ConvWT(x_list[i]).permute(0, 2, 1)

        return x_list


    def ConvWT(self, x):
        B, C, L = x.shape
        x_wt = x.unsqueeze(3)
        h, w = self.find_closest_factors(L)
        x_wt = x_wt.reshape(
            B, C, h, w)
        x_wt = self.down_wt(x_wt)
        x_wt = x_wt.permute(0, 2, 1)
        return x_wt

    def find_closest_factors(self, L):
        # 计算平方根
        sqrt_n = int(math.sqrt(L))

        # 尝试找出最接近的两个因数
        for i in range(sqrt_n, 0, -1):
            if L % i == 0:
                factor1 = i
                factor2 = L // i
                return factor1, factor2

    def classification(self, x, x_mark_enc):
        B, L, C = x.shape
        # RIN
        # 使其均值为0
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        # print(x_var)
        x = x / torch.sqrt(x_var)

        # 卷积变换，提升正样本准确率
        # -----------------------------------------------
        x_list = self.MulitProcessInput(x)

        # 卷积变换-----------------------
        #小波变换结合fft 效果提升
        x_wt_list = self.MulitConvWT(x_list)

        x_wt = torch.concatenate(x_wt_list, dim=2)

        wt_specxy = self.WTLinear(x_wt).permute(0, 2, 1)

        low_specx = torch.fft.rfft(x, dim=1)

        low_specx[:, self.dominance_freq:] = 0  # LPF
        low_specx = low_specx[:, 0:self.dominance_freq, :]  # LPF


        # print(low_specx.permute(0,2,1))
        low_specxy_ = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(0, 2, 1)

        # print(low_specxy_)
        P_B, P_L, P_C = low_specxy_.size(0), int(self.seq_len / 2 + 1), low_specxy_.size(2)

        low_specxy = torch.zeros([P_B, P_L, P_C], dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:, 0:low_specxy_.size(1), :] = low_specxy_  # zero padding

        # wt change
        low_specxy = low_specxy + wt_specxy

        low_xy = torch.fft.irfft(low_specxy, dim=1)


        # xy = low_xy
        xy = (low_xy) * torch.sqrt(x_var) + x_mean
        output = xy.reshape(xy.shape[0], -1)
        output = self.my_projection(output) # (batch_size, num_classes)



        return output

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        B, L, C = x_enc.shape
        # RIN
        # 使其均值为0
        x_mean = torch.mean(x_enc, dim=1, keepdim=True)
        x = x_enc - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        # print(x_var)
        x = x / torch.sqrt(x_var)

        # 卷积变换，提升正样本准确率
        # -----------------------------------------------
        x_list = self.MulitProcessInput(x)
        # 卷积变换-----------------------
        # 小波变换结合fft 效果提升
        x_wt_list = self.MulitConvWT(x_list)
        x_wt = torch.concatenate(x_wt_list, dim=2)
        wt_specxy = self.WTLinear(x_wt).permute(0, 2, 1)

        #FED
        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:, self.dominance_freq:] = 0  # LPF
        low_specx = low_specx[:, 0:self.dominance_freq, :]  # LPF

        if self.individual:
            low_specxy_ = torch.zeros(
                [low_specx.size(0), int(self.dominance_freq * self.length_ratio), low_specx.size(2)],
                dtype=low_specx.dtype).to(low_specx.device)
            for i in range(self.channels):
                low_specxy_[:, :, i] = self.freq_upsampler[i](low_specx[:, :, i].permute(0, 1)).permute(0, 1)
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(0, 2, 1)


        P_B, P_L, P_C = low_specxy_.size(0), int((self.seq_len + self.pred_len) / 2 + 1), low_specxy_.size(2)
        low_specxy = torch.zeros([P_B, P_L, P_C], dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:, 0:low_specxy_.size(1), :] = low_specxy_  # zero padding

        # MWC FED Mixing
        low_specxy = low_specxy + wt_specxy

        low_xy = torch.fft.irfft(low_specxy, dim=1)

        low_xy = low_xy * self.length_ratio  # energy compemsation for the length change

        xy = (low_xy) * torch.sqrt(x_var) + x_mean
        return xy

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
