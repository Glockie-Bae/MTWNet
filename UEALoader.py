# -*- coding: utf-8 -*-
# @File : UEA数据处理.py
import pandas as pd
import os
from scipy.io import arff
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_data(data_path):
    data, meta = arff.loadarff(data_path)
    res_data = []
    res_labels = []
    for t_data, t_label in data:
        t_data = np.array([d.tolist() for d in t_data])
        t_label = t_label.decode("utf-8")
        res_data.append(t_data)
        res_labels.append(t_label)
    return np.array(res_data).swapaxes(1, 2), np.array(res_labels)


def save_csv_data(data, file_name, tag='train'):
    channel_num = data.shape[0]
    for channel_idx in tqdm(range(channel_num)):  # 读取每一个通道的数据
        channel_data = data[channel_idx]
        # 将 NumPy 数组转换为 pandas DataFrame
        df = pd.DataFrame(channel_data)
        data_file = f'{file_name}\\{tag}_dim{channel_idx + 1}.xlsx'
        df.to_excel(data_file, index=False)
        # df.to_excel(data_file, index=False,header=None)#第一行不添加时间戳


def save_data(train_data, train_label, test_data, test_label, file_name, data_name, tag='npz'):
    if tag == 'npz':
        # np.savez(file_name,train_X=train_data,train_Y=train_label,test_X=test_data,test_Y=test_label)
        np.savez_compressed(file_name, train_X=train_data, train_Y=train_label, test_X=test_data,
                            test_Y=test_label)  # 使用压缩方法
    elif tag == 'npy':
        np.save(target_path + f"\\{data_name}\\train_data", train_data)
        np.save(target_path + f"\\{data_name}\\train_labels", train_label)
        np.save(target_path + f"\\{data_name}\\test_data", test_data)
        np.save(target_path + f"\\{data_name}\\test_labels", test_label)
    elif tag == 'xlsx':
        # 将通道置于第一个维度 维度变为N L C -> C N L
        train_data = np.transpose(train_data, (2, 0, 1))
        test_data = np.transpose(test_data, (2, 0, 1))
        # 将每个通道数据写入1个xlsx文件中
        save_csv_data(train_data, file_name, tag='train')
        save_csv_data(test_data, file_name, tag='test')
        # 写入标签,由于标签可能不是数字，所以不写入csv文件而是excel文件
        df_train_label = pd.DataFrame(train_label)
        df_train_label.to_excel(f'{file_name}\\train_label.xlsx', index=False, header=None)

        df_test_label = pd.DataFrame(test_label)
        df_test_label.to_excel(f'{file_name}\\test_label.xlsx', index=False, header=None)
    elif tag == "ts":
        train_path = filepath + "OverSampleData_TRAIN.ts"
        test_path = filepath + "OverSampleData_TEST.ts"

        data2ts(train_data, train_lablels, os.path.join(path, train_path), addition_info)
        data2ts(test_data, test_labels, os.path.join(path, test_path), addition_info)
    else:
        print('No implemented...')


arff_path = 'F:\\_Sorrow\\SCNU_M\\数据\\DotDataset\\Multivariate_arff'
tag = 'npy'  # 保存的数据的格式
target_path = f'F:\\_Sorrow\\SCNU_M\\数据\\DotDataset\\Multivariate_arff_{tag}'


def process_uea_data(arff_path, target_path, tag):
    # 如果数据目录不存在，则创建
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    wrong_data = ['EigenWorms']  # 由于数据集太大，处理成xlsx内存不够
    for data_name in os.listdir(arff_path):
        train_file = f'{arff_path}\\{data_name}\\{data_name}_TRAIN.arff'
        test_file = f'{arff_path}\\{data_name}\\{data_name}_TEST.arff'
        train_data, train_label = load_data(train_file)
        test_data, test_label = load_data(test_file)
        data_dir = f'{target_path}\\{data_name}'
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        if tag == 'npz':
            file_name = f'{data_dir}\\{data_name}.npz'
            save_data(train_data, train_label, test_data, test_label, file_name, data_name, tag=tag)
        elif tag == 'npy':
            file_name = f'{data_dir}\\{data_name}.npy'


            save_data(train_data, train_label, test_data, test_label, file_name, data_name, tag=tag)
        elif tag == 'xlsx':
            # EigenWorms数据序列长度太大了，处理不了
            if data_name in wrong_data:
                continue
            file_name = f'{data_dir}'
            save_data(train_data, train_label, test_data, test_label, file_name, data_name, tag=tag)
        else:
            print('No implemented...')
        print_info = f'{data_name} finished!'
        print(f'{print_info}{(70 - len(print_info)) * "="}')


#process_uea_data(arff_path, target_path, tag)
#---------------------------读取数据-------------------------------




# data_path = "F:\\_Sorrow\\SCNU_M\\数据\\DotDataset\\Multivariate_arff_npy\\MotorImagery\\"
# data_npy = data_path + "train_data.npy"
# #加载npz数据
# train_data = np.load(data_npy, allow_pickle=True)
# print(train_data.shape)
#
# train_labels_path = data_path + "train_labels.npy"
# train_labels = np.load(train_labels_path, allow_pickle=True)
# for index, item in enumerate(train_labels):
#     if item == "finger":
#         train_labels[index] = 0
#     if item == "tongue":
#         train_labels[index] = 1
#
# test_labels_path = data_path + "test_labels.npy"
# test_labels = np.load(test_labels_path, allow_pickle=True)
# for index, item in enumerate(test_labels):
#     if item == "finger":
#         test_labels[index] = 0
#     if item == "tongue":
#         test_labels[index] = 1
#
# print(train_labels)
# print(test_labels)
# np.save(train_labels_path, train_labels)
# np.save(test_labels_path, test_labels)

