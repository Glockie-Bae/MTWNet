import torch
import numpy as np
import argparse
import time
from uea import subsample, interpolate_missing, Normalizer, Normalizer_xi, collate_fn_relation
from load import UEAloader_xi5
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from math import sqrt
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error

def percentage_accuracy(true, pred):
    mape = sum(abs((t - p) / t) for t, p in zip(true, pred) if t != 0) / len(true)
    accuracy = 100 - mape * 100
    return accuracy

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='TSER', help='task') # ['classification', 'long_term_forecast','TSER']
parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
parser.add_argument('--check_path', type=str, default='./trial_checkpoint', help='task')
parser.add_argument('--dataset_path', type=str, default="F:\_Sorrow\SCNU_M\数据\DotDataset\public_dataset\Multivariate_ts\\", help='task dataset')
parser.add_argument('--dataset_name', type=str, default="AtrialFibrillation", help='task dataset')
parser.add_argument('--public_data', type=bool, default=False)
parser.add_argument('--classification', type=str, default="TSER", help='["Binary", "Multi", "TSER", long_term_forecast]')
parser.add_argument('--dataset', type=str, default="F:\_Sorrow\SCNU_M\数据\DotDataset\DotDateset\\all\\", help='task dataset')
parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')


parser.add_argument('--model', type=str,default='MFC_v3' , help='model name in directory ./models')

parser.add_argument('--freq', type=str, default='s')
parser.add_argument('--channels', type=int, default=3,       help='for channel')

# model define
parser.add_argument('--top_k', type=int, default=3,       help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=3, help='for Inception')

parser.add_argument('--enc_in', type=int, default=6,      help='encoder input size')

# 8 16 32 128
# 3
parser.add_argument('--d_model', type=int, default=16,   help='dimension of model')
parser.add_argument('--n_heads', type=int, default=3,     help='num of heads')

parser.add_argument('--e_layers', type=int, default=2,    help='num of encoder layers')

# 32
parser.add_argument('--d_ff', type=int, default=32,     help='dimension of fcn')

parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
# 2
parser.add_argument('--factor', type=int, default=2,      help='attn factor')

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0,          help='gpu')

# SegRNN
parser.add_argument('--rnn_type', default='gru', help='rnn_type')
parser.add_argument('--dec_way', default='pmf', help='decode way')

parser.add_argument('--seg_len', type=int, default=3600, help='segment length')
parser.add_argument('--win_len', type=int, default=48, help='windows length')
parser.add_argument('--pred_len', type=int, default=3600, help='windows length')


parser.add_argument('--channel_id', type=int, default=1, help='Whether to enable channel position encoding')
parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
parser.add_argument('--is_FFT', type=int, default=0, help='whether need to embed fre data; True 1 False 0')

# TimeMixer
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_layers', type=int, default=1, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=10, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default="avg",
                        help='down sampling method, only support avg, max, conv')
parser.add_argument('--channel_independence', type=int, default=0,
                        help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')

#ModernTCN
parser.add_argument('--stem_ratio', type=int, default=6, help='stem ratio')
parser.add_argument('--downsample_ratio', type=int, default=2, help='downsample_ratio')
parser.add_argument('--ffn_ratio', type=int, default=2, help='ffn_ratio')
parser.add_argument('--patch_size', type=int, default=16, help='the patch size')
parser.add_argument('--patch_stride', type=int, default=8, help='the patch stride')

parser.add_argument('--num_blocks', nargs='+',type=int, default=[1,1,1,1], help='num_blocks in each stage')
parser.add_argument('--large_size', nargs='+',type=int, default=[31,29,27,13], help='big kernel size')
#parser.add_argument('--large_size', nargs='+',type=int, default=[5,5,5,5], help='big kernel size')
parser.add_argument('--small_size', nargs='+',type=int, default=[3,3,3,3], help='small kernel size for structral reparam')
parser.add_argument('--dims', nargs='+',type=int, default=[16,16,16,16], help='dmodels in each stage')
parser.add_argument('--dw_dims', nargs='+',type=int, default=[32,32,32,32], help='dw dims in dw conv in each stage')



#GCN
parser.add_argument('--gcn_depth', type=int, default=2, help='')
parser.add_argument('--propalpha', type=float, default=0.3, help='')
parser.add_argument('--conv_channel', type=int, default=8, help='')
parser.add_argument('--skip_channel', type=int, default=8, help='')
parser.add_argument('--c_out', type=int, default=6, help='')

#Dlinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

parser.add_argument('--dropout', type=float, default=0.1,  help='dropout')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')



parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

parser.add_argument('--epoch_count', type=int, default=30,
                        help='the maximum count of the epoch in training')

parser.add_argument('--loss_fn', type=str, default="CrossEntropyLoss",
                        help='CrossEntropyLoss or FocalLoss of FocalLoss_a')
parser.add_argument('--num_class', type=int, default=2,
                        help='num_class')

parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')

parser.add_argument('--gamma', type=float, default=2)
parser.add_argument('--alpha', type=float, default=0.33)
parser.add_argument('--reduction', type=str, default='mean')
parser.add_argument('--get_MoE_para', type=bool, default=True)

args = parser.parse_args(args=[])

args.device = f"GPU:{args.gpu}"
if torch.cuda.is_available():
    args.device = f"cuda:{args.gpu}"
print(f"use device: {args.device}")

#args.device = f"cpu"
args.label_len = 2
args.pred_len = 0
args.seq_len = 3600


num_workers = 0

# from models.XiNet6_splitMK import Model as XiNet
# from models.TimesNet import  Model as TimesNet
# from models.FITS import Model as FITS
from models.MFC_v3 import Model as MFC_v3
# from models.TSMixer import Model as TSMixer
# from models.TimeMixer import Model as TimeMixer
import os

#mdl1 = XiNet(args)
#mdl1 = TimesNet(args)

#mdl1.load_state_dict(torch.load("../TimesNet/checkpoint_46.pth", map_location=torch.device(args.device)))
#35


data_filepath = os.path.join(args.dataset, args.classification, "train" + "_data.npy")
labels_filepath = os.path.join(args.dataset, args.classification, "train" + "_labels.npy")

data = np.load(data_filepath).transpose(0, 2, 1) # [size, seq_len, feat_num]
labels = np.load(labels_filepath)


data_size = data.shape[0]
seq_len = data.shape[1]
feat_num = data.shape[2]
data = data.reshape(-1, feat_num)


# for name in par_dict:
#     parameter = par_dict[name]
#     print(name, parameter.numpy().shape)



def predict_binary_all():
    test_ds = UEAloader_xi5(
        dataset_path=args.dataset,
        dataset_name='classification',
        flag="test",
        public_data=False,
        classification="classification"
    )

    args.enc_in = test_ds.data.shape[2]

    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=lambda x: collate_fn_relation(x, max_len=args.seq_len)
    )


    for index in range(2):
        args.task_name = 'classification'
        mdl1 = MFC_v3(args).cuda(args.gpu)
        mdl1.load_state_dict(torch.load(f"./trial_checkpoint/checkpoint_{index}.pth",
                                        map_location=torch.device(args.device)))

        test_correct = 0
        test_total = 0
        test_total_T = 0
        test_total_TP = 0
        test_total_TN = 0
        test_total_F = 0
        test_total_FP = 0
        test_total_FN = 0
        test_running_loss = 0
        test_recall = []
        test_f1 = []
        time_list = []
        # 模型进入验证状态
        mdl1.eval()
        with torch.no_grad():
            for x, y, padding_mask in test_dl:
                # 与训练状态一致
                x = x.to(args.device)
                y = y.reshape(-1).long().to(args.device)
                padding_mask = padding_mask.to(args.device)
                # corr_data = corr_data.to(device)
                start_time = time.time()
                y_pred = mdl1(x, padding_mask, None, None, None)
                end_time = time.time()
                time_list.append(end_time - start_time)
                # loss = loss_fn(y_pred, y)
                y_pred = torch.argmax(y_pred, dim=1)

                # 返回一个新的tensor，从当前步骤计算图中分离出来
                trues = y.detach().cpu()
                preds = y_pred.detach().cpu()

                test_correct += (y_pred == y).sum().item()
                for i, item in enumerate(y):
                    # label true
                    if item == 1:
                        test_total_T += 1
                        # label true pred true
                        if y_pred[i] == 1:
                            test_total_TP += 1
                        # label true pred false
                        else:
                            test_total_FN += 1

                    # label false
                    if item == 0:
                        test_total_F += 1
                        # label false pred false
                        if y_pred[i] == 0:
                            test_total_TN += 1
                        # label false pred true
                        else:
                            test_total_FP += 1
                test_total += y.size(0)

        test_acc = test_correct / test_total
        test_T_acc = test_total_TP / test_total_T
        test_F_acc = test_total_TN / test_total_F
        precision = test_total_TP / (test_total_TP + test_total_FP)
        recall = test_total_TP / (test_total_TP + test_total_FN)
        avg_time = np.mean(time_list) * 1000
        print(f"avg time : {format(avg_time, '.3f')} ms")
        print(f"index : {index}")
        print("test_acc: ", format(test_acc, '.3f'))
        print("test_T_acc: ", format(test_T_acc, '.3f'))
        print("test_F_acc: ", format(test_F_acc, '.3f'))
        print("test_precision", format(precision, '.3f'))
        print("test_recall", format(recall, '.3f'))
        print("test_f1", format(2 * (precision * recall) / (precision + recall), '.3f'))
        print(
            "----------------------------------------------------------------------------------------------------------")

def predict_TSER_best(index, threshold = 7500):
    test_ds = UEAloader_xi5(
        dataset_path=args.dataset,
        dataset_name='TSER',
        flag="test",
        public_data=False,
        classification="TSER"
    )

    args.enc_in = test_ds.data.shape[2]

    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=lambda x: collate_fn_relation(x, max_len=args.seq_len)
    )
    args.task_name = 'TSER'
    mdl1 = MFC_v3(args).cuda(args.gpu)
    mdl1.load_state_dict(torch.load(f"./trial_checkpoint/checkpoint_{index}.pth",
                                    map_location=torch.device(args.device)))

    test_correct = 0
    test_total = 0
    test_total_T = 0
    test_total_TP = 0
    test_total_TN = 0
    test_total_F = 0
    test_total_FP = 0
    test_total_FN = 0

    all_conv_scores = []  # 用于保存所有 batch 的 conv_score
    all_fm_scores = []  # 用于保存所有 batch 的 fm_score

    # 模型进入验证状态
    mdl1.eval()
    with torch.no_grad():
        true_list = []
        pred_list = []
        for x, y, padding_mask in test_dl:
            # 与训练状态一致
            x = x.to(args.device)
            y = y.reshape(-1).long().to(args.device)
            padding_mask = padding_mask.to(args.device)
            # corr_data = corr_data.to(device)

            y_pred, conv_scores, fm_scores = mdl1(x, padding_mask, None, None, None)
            # 统计所有batch的热力图特征
            for i, (cscore, fscore) in enumerate(zip(conv_scores, fm_scores)):
                # batch 内求平均
                # 保存到列表（按层存储）
                if len(all_conv_scores) <= i:
                    all_conv_scores.append([cscore])
                    all_fm_scores.append([fscore])
                else:
                    all_conv_scores[i].append(cscore)
                    all_fm_scores[i].append(fscore)

            for i, item in enumerate(y):
                # label true
                item = int(item)

                print(f" y is {[item]}, pred_y is {[int(y_pred[i])]}", end="; ")
                true_list.append(item)
                pred_list.append(int(y_pred[i]))
                if item >= threshold:
                    test_total_T += 1
                    # label true pred true
                    if y_pred[i] >= threshold:
                        test_total_TP += 1
                    # label true pred false
                    else:
                        test_total_FN += 1

                # label false
                if item < threshold:
                    test_total_F += 1
                    # label false pred false
                    if y_pred[i] < threshold:
                        test_total_TN += 1
                    # label false pred true
                    else:
                        test_total_FP += 1
            print("\n")
            test_total += y.size(0)

    # 对每层所有 batch 再求平均 → 测试集平均
    avg_conv_scores = []  # 用于保存每个尺度的平均图
    avg_fm_scores = []
    for conv_scale_scores in all_conv_scores:
        # scale_scores: list of 60 tensors, each (6,3)
        # 先堆叠成 (60,6,3)
        conv_array = np.stack(conv_scale_scores, axis=0)
        # 对第0维求平均 -> (6,3)
        conv_scale_mean = conv_array.mean(axis=0)
        avg_conv_scores.append(conv_scale_mean)

    for fm_scale_scores in all_fm_scores:
        # scale_scores: list of 60 tensors, each (6,3)
        # 先堆叠成 (60,6,3)
        fm_array = np.stack(fm_scale_scores, axis=0)
        # 对第0维求平均 -> (6,3)
        fm_scale_mean = fm_array.mean(axis=0)
        avg_fm_scores.append(fm_scale_mean)


    # 绘制大图：2行，num_layers列
    num_layers = len(avg_conv_scores)
    plt.figure(figsize=(5 * num_layers, 10))

    for i in range(num_layers):
        # 卷积核权重
        plt.subplot(2, num_layers, i + 1)
        sns.heatmap(avg_conv_scores[i], annot=True, cmap='viridis')
        plt.xlabel('Conv kernels (1x1,3x3,5x5)')
        plt.ylabel('Channels')
        plt.title(f'Scale {i} - Avg Conv Weights')

        # 小波权重
        plt.subplot(2, num_layers, i + 1 + num_layers)
        sns.heatmap(avg_fm_scores[i], annot=True, cmap='viridis')
        plt.xlabel('Wavelet features')
        plt.ylabel('Channels')
        plt.title(f'Scale {i} - Avg Wavelet Weights')

    plt.tight_layout()
    plt.show()

    per_acc = percentage_accuracy(true_list, pred_list)
    mae = mean_absolute_error(true_list, pred_list)
    mse = mean_squared_error(true_list, pred_list)
    rmse = sqrt(mse)
    test_acc = (test_total_TP + test_total_TN) / test_total
    test_T_acc = test_total_TP / test_total_T
    test_F_acc = test_total_TN / test_total_F
    precision = test_total_TP / (test_total_TP + test_total_FP)
    recall = test_total_TP / (test_total_TP + test_total_FN)
    print(f"index : {index}")
    print("test_acc: ", format(test_acc, '.3f'))
    print("test_T_acc: ", format(test_T_acc, '.3f'))
    print("test_F_acc: ", format(test_F_acc, '.3f'))
    print("test_precision", format(precision, '.3f'))
    print("test_recall", format(recall, '.3f'))
    print("test_f1", format(2 * (precision * recall) / (precision + recall), '.3f'))
    print("test_MAE", format(mae, '.3f'))
    print("percentage_accuracy", format(per_acc, '.3f'))
    print("test_RMSE", format(rmse, '.3f'))
    print(
        "----------------------------------------------------------------------------------------------------------")

import Orange

if __name__ == "__main__":
    predict_TSER_best(8, 7500)
    #predict_TSER_best(1, 7500)
    # import matplotlib.pyplot as plt
    #
    # import matplotlib
    # matplotlib.use('TkAgg')  # 不显示图则加上这两行
    # names = ['MLP', 'FCN', 'InceptionTime', 'TapNet', 'CMFM+SVM', 'MiniROCKET', 'Conv-GRU', 'DA-NET', 'Times-Net', 'OS-CNN', 'XEM', 'Ours']
    # avranks = [8.897, 6.931, 6.276, 5.000, 6.620, 4.103, 4.552, 5.793, 4.897, 4.120, 3.828, 3.345]
    # datasets_num = 29
    # CD = Orange.evaluation.scoring.compute_CD(avranks, datasets_num, alpha='0.05', test='nemenyi')
    # Orange.evaluation.scoring.graph_ranks(avranks, names, cd=CD, width=8, textspace=1.5, reverse=True)
    # plt.show()



