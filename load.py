from sklearn.metrics import normalized_mutual_info_score
import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from uea import subsample, interpolate_missing, Normalizer, Normalizer_xi
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
import utils.TSC_multivariate_data_loader as TSC_loader
from utils.TSER_data_loader import TSER_data_loader

import tqdm

warnings.filterwarnings('ignore')

class UEAloader_xi5(Dataset):
    def __init__(self, dataset_path, dataset_name, flag, load_path=None, public_data = False, classification = None, task_name=None):
        self.root_path = dataset_path
        data_filepath = os.path.join(dataset_path, dataset_name, flag + "_data.npy")
        labels_filepath = os.path.join(dataset_path, dataset_name, flag + "_labels.npy")

        if public_data == False:
            self.data = np.load(data_filepath)
            self.labels = np.load(labels_filepath)

        else:
            if classification == "TSER":
                X_train, y_train, X_test, y_test = TSER_data_loader(dataset_path, dataset_name)
                X_train = X_train.transpose(0, 2, 1)
                X_test = X_test.transpose(0, 2, 1)
            else:
                X_train, y_train, X_test, y_test = TSC_loader.TSC_multivariate_data_loader(dataset_path, dataset_name)

            if flag == "train":
                self.data = X_train
                self.labels = y_train
            elif flag == "test":
                self.data = X_test
                self.labels = y_test
        
        self.data = self.data.transpose(0, 2, 1)  # [size, seq_len, feat_num]
        normalize = Normalizer_xi()
        if public_data != True:
            self.data = normalize.normalize_data(self.data)

        self.labels = self.labels
        
        self.data_size = self.data.shape[0]
        self.seq_len = self.data.shape[1]
        self.feat_num = self.data.shape[2]





    def instance_norm(self, case):
        # special process for numerical stability
        if self.root_path.count('EthanolConcentration') > 0:
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(
                torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        return self.instance_norm(torch.from_numpy(self.data[ind])), \
            torch.tensor([self.labels[ind]])#, self.data_corr[ind]

    def __len__(self):
        return len(self.labels)


class UEAloader_xi5s(Dataset):
    def __init__(self, root_path, flag):
        self.root_path = root_path
        data_filepath = os.path.join(root_path, flag + "_data.npy")
        labels_filepath = os.path.join(root_path, flag + "_labels.npy")
        # data: [size, feat_num, seq_len]
        # labels: [size]

        self.data = np.load(data_filepath)
        self.labels = np.load(labels_filepath)
        # data: [size, feat_num, seq_len]
        # labels: [size]
        
        self.data = self.data.transpose(0, 2, 1)  # [size, seq_len, feat_num]
        self.Normalizer = Normalizer_xi()
        self.data = self.Normalizer.normalize_data(self.data)

        self.labels = self.labels
        
        self.data_size = self.data.shape[0]
        self.seq_len = self.data.shape[1]
        self.feat_num = self.data.shape[2]
        
    def instance_norm(self, case):
        # special process for numerical stability
        if self.root_path.count('EthanolConcentration') > 0:
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(
                torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        return self.instance_norm(torch.from_numpy(self.data[ind])), \
            torch.tensor([self.labels[ind]])

    def __len__(self):
        return len(self.labels)


