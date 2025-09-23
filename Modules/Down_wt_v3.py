import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
import torch.nn.functional as F
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import torchvision
class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch, seq_len=None):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar') #mexh
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.conv_bn_relu2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.conv_bn_relu3 = nn.Sequential(
            nn.Conv2d(in_ch , out_ch, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.mix = nn.Linear(3 * out_ch, out_ch)

        # 对卷积核MoE
        self.conv_gate = nn.Linear(seq_len, 3)

        # gate 层：给四个 DWT 特征图加权
        self.fm_gate = nn.Linear(seq_len, 4)  # 对每个通道生成 4 个权重
        self.conv_temperature = 4.0
        self.fm_temperature = 4.0


    def forward(self, x):
        B, C, W, H = x.shape

        # 构建Conv三个卷积核的专家
        conv_score = F.softmax(self.conv_gate(x.reshape(B * C, -1) / self.conv_temperature), dim=-1)

        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]

        # 构建小波四个特征图专家
        experts = [yL, y_HL, y_LH, y_HH]
        expert_stack = torch.stack(experts, dim=-1)  # (B,C,H',W',4)

        # ----------------- Wavelet feature maps Moe -----------------
        # reshape到 (B*C, H'*W') 以便做线性层
        B, C, Hf, Wf, E = expert_stack.shape
        expert_flat = expert_stack.view(B * C, Hf * Wf, E)  # (B*C, L, 4)

        # gate 输出 (B*C, 4)
        fm_score = F.softmax(self.fm_gate(expert_flat.reshape(B*C,-1)[:,:W * H]) / self.fm_temperature, dim=-1)  # (B*C, 4)

        # ----------------- MoE加权 -----------------
        # 对最后一维（4个专家）加权
        output_flat = torch.einsum("BLE,BE->BL", expert_flat, fm_score)  # (B*C, L)
        x = output_flat.view(B, C, Hf, Wf)

        # x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x1 = self.conv_bn_relu(x)
        x1 = x1.reshape(B * C, -1)

        x2 = self.conv_bn_relu2(x)
        x2 = x2.reshape(B * C, -1)

        x3 = self.conv_bn_relu3(x)
        x3 = x3.reshape(B * C, -1)

        conv_outputs = torch.stack([x1, x2, x3], dim=-1)
        output = torch.einsum("BLE,BE->BL", conv_outputs, conv_score).reshape(B , C, -1)


        conv_score = conv_score.reshape(B, C, 3)  # 这里多返回score
        fm_score = fm_score.reshape(B, C, 4)  # 这里多返回score

        # 对整个 batch 取平均
        conv_score_mean = conv_score.detach().cpu().mean(dim=0).numpy()  # shape (C, 3)
        fm_score_mean = fm_score.detach().cpu().mean(dim=0).numpy()  # shape (C, 4)

        # plt.figure(figsize=(16, 6))
        #
        # # 第一张子图：卷积尺度平均权重
        # plt.subplot(1, 2, 1)
        # sns.heatmap(conv_score_mean, annot=True, cmap='viridis')
        # plt.xlabel('Features (1×1,3×3,5×5)')
        # plt.ylabel('Channels')
        # plt.title('Average Conv kernel weights (All batches)')
        #
        # # 第二张子图：小波特征平均权重
        # plt.subplot(1, 2, 2)
        # sns.heatmap(fm_score_mean, annot=True, cmap='viridis')
        # plt.xlabel('Wavelet features')
        # plt.ylabel('Channels')
        # plt.title('Average Wavelet feature weights (All batches)')
        #
        # plt.tight_layout()
        # plt.show()
        return output, conv_score_mean, fm_score_mean

def draw():
    pass
    # plt.figure(figsize=(20, 18))
    # plt.subplot(2, 2, 1)
    # img = x[0][0].cpu().numpy()
    # img = img.astype('float32')
    # img = (img - np.min(img)) / (np.max(img) - np.min(img))
    #
    # im = plt.imshow(img, cmap='Blues', interpolation='nearest')
    # cbar = plt.colorbar(im)  # 获取颜色条对象
    # cbar.set_label('Value Bar', fontsize=20, fontweight='bold')  # 给颜色条加上文本说明
    # plt.title('2D Input', fontsize=20, fontweight='bold')
    #
    # plt.subplot(2, 2, 2)
    # img_HL = y_HL[0][0].cpu().numpy()
    # img_HL = img_HL.astype('float32')
    # img_HL = (img_HL - np.min(img_HL)) / (np.max(img_HL) - np.min(img_HL))
    # plt.imshow(img_HL, cmap='Blues', interpolation='nearest')
    # plt.colorbar()  # 显示颜色条
    # plt.title('High-Low', fontsize=20, fontweight='bold')
    #
    # plt.subplot(2, 2, 3)
    # img_LH = y_HL[0][0].cpu().numpy()
    # img_LH = img_LH.astype('float32')
    # img_LH = (img_LH - np.min(img_LH)) / (np.max(img_LH) - np.min(img_LH))
    # plt.imshow(img_LH, cmap='Blues', interpolation='nearest')
    # plt.colorbar()  # 显示颜色条
    # plt.title('Low-High', fontsize=20, fontweight='bold')
    #
    # plt.subplot(2, 2, 4)
    # img_HH = y_HL[0][0].cpu().numpy()
    # img_HH = img_HH.astype('float32')
    # img_HH = (img_HH - np.min(img_HH)) / (np.max(img_HH) - np.min(img_HH))
    # plt.imshow(img_HH, cmap='Blues', interpolation='nearest')
    # plt.colorbar()  # 显示颜色条
    # plt.title('High-High', fontsize=20, fontweight='bold')
    #
    # plt.savefig('tmp.pdf', bbox_inches='tight', dpi=500)  # 保存成PDF放大后不失真（默认保存在了当前文件夹下）
    # plt.show()

    # 加一个文件夹

if __name__ == '__main__':
    block = Down_wt(64, 128)  # 输入通道数，输出通道数
    input = torch.rand(3, 64, 64, 64)  # 输入B C H W
    output = block(input)
    print(output.size())