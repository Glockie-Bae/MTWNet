import os
import torch
import importlib
from utils.str2bool import str2bool



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='long_term_forecast', help='task') # ['classification', 'long_term_forecast']
parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
parser.add_argument('--check_path', type=str, default='./trial_checkpoint', help='task')
parser.add_argument('--dataset_path', type=str, default="F:\_Sorrow\SCNU_M\数据\DotDataset\public_dataset\Multivariate_ts\\", help='task dataset')
parser.add_argument('--dataset_name', type=str, default="AtrialFibrillation", help='task dataset')
parser.add_argument('--public_data', type=bool, default=True)
parser.add_argument('--classification', type=str, default="long_term_forecast", help='["Binary", "Multi", "TSER", long_term_forecast]')
parser.add_argument('--dataset', type=str, default="F:\_Sorrow\SCNU_M\数据\DotDataset\DotDateset\\all\\", help='task dataset')
parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')


parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETT-small/ETTh1.csv', help='data file')
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


# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=192, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

#Augmentation
parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")

#optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')


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

parser.add_argument('--small_kernel_merged', type=str2bool, default=False, help='small_kernel has already merged or not')
parser.add_argument('--call_structural_reparam', type=bool, default=False, help='structural_reparam after training')
parser.add_argument('--use_multi_scale', type=str2bool, default=True, help='use_multi_scale fusion')

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

#----------------------train epoch --------------------------
parser.add_argument('--epoch_count', type=int, default=10,
                        help='the maximum count of the epoch in training')
parser.add_argument('--trial', type=int, default=10,
                        help='the maximum count of the epoch in all training')


parser.add_argument('--loss_fn', type=str, default="CrossEntropyLoss",
                        help='CrossEntropyLoss or FocalLoss of FocalLoss_a')
parser.add_argument('--num_class', type=int, default=2,
                        help='num_class')

parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')

args = parser.parse_args()
args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.use_gpu else "cpu"
print(f"use cuda:{args.gpu}")
print(f"save check_point path is {args.check_path}")
print(f"dataset path is {args.dataset_path}")

if not os.path.exists(args.check_path):
    print(f"make dir: {args.check_path}")
    os.mkdir(os.path.join(args.check_path))






if __name__ == "__main__":

    model_list = ["MFC_v3"]
    print(f"Train Mode : {args.classification}")
    for model in model_list:

        args.model = model
        print(f"Start Model : {model}")
        Model = importlib.import_module(f'models.{args.model}').Model

        if args.classification == "TSER":
            if args.public_data == True:
                import sys

                sys.path.append("./code/")
                from DataLoader.TSER_public import TSER_Fit

                a = TSER_Fit(args, Model)
                a.Fit()
            else:
                import sys

                sys.path.append("./code/")
                from DataLoader.TSER_Dot import TSER_Fit

                a = TSER_Fit(args, Model)
                a.Fit()
        elif args.classification == "long_term_forecast":
            import sys

            sys.path.append("./code/")
            from DataLoader.TSF_public import TSF_Fit

            a = TSF_Fit(args, Model)
            a.Fit()

        else:
            if args.public_data == True:
                import sys

                sys.path.append("./code/")
                from DataLoader.TSC_public import TSC_Fit

                a = TSC_Fit(args, Model)
                a.Fit()

            elif args.public_data == False:
                import sys

                sys.path.append("./code/")
                from DataLoader.TSC_Dot import TSC_Fit

                a = TSC_Fit(args, Model)
                a.Fit()








