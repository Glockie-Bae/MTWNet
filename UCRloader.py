import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


root_path = "../DotDataset/UCRArchive_2018/FordA/"

x_train, y_train = readucr(root_path + "FordA_TRAIN.tsv")
x_test, y_test = readucr(root_path + "FordA_TEST.tsv")

classes = np.unique(np.concatenate((y_train, y_test), axis=0))

def show_data(x_data):
    plt.figure()
    for c in classes:
        c_x_train = x_train[y_train == c]
        plt.plot(c_x_train[0], label="class " + str(c))
    plt.legend(loc="best")
    plt.show()
    plt.close()

def save_data(np_save_path, train_data, train_labels, test_data, test_labels):
    print(np_save_path)
    if not os.path.exists(np_save_path):
        print("make file : ", np_save_path)
        os.mkdir(np_save_path)


    np.save(os.path.join(np_save_path, "train_data.npy"), train_data)
    np.save(os.path.join(np_save_path, "train_labels.npy"), train_labels)
    np.save(os.path.join(np_save_path, "test_data.npy"), test_data)
    np.save(os.path.join(np_save_path, "test_labels.npy"), test_labels)


x_train = x_train.reshape((x_train.shape[0], 1,  x_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0],1, x_test.shape[1]))

num_classes = len(np.unique(y_train))


from sklearn.preprocessing import StandardScaler, RobustScaler
# StandardScaler类是一个用来讲数据进行归一化和标准化的类
# fit_transform方法是fit和transform的结合，fit_transform(X_train) 意思是找出X_train的平均值和标准差，并应用在X_train上。
# 这时对于X_test，我们就可以直接使用transform方法。因为此时StandardScaler已经保存了X_train的平均值和标准差。

idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]


y_train[y_train == -1] = 0
y_test[y_test == -1] = 0

# 转换数据类型
train_data = x_train.astype(np.float32)
test_data = x_test.astype(np.float32)

train_labels = y_train.astype(np.float32)
test_labels = y_test.astype(np.float32)

# 保存数据
np_save_path = f"../DotDataset/DotDateset/UCR"
save_data(np_save_path, train_data, train_labels, test_data, test_labels)

