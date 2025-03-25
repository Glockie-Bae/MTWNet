import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import torch.nn.functional as F
import tqdm

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score


class EarlyStopping:
    def __init__(self, trial_id, patience=7, verbose=False, delta=0):
        self.trial_id = trial_id
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score_muti = None

        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, score_muti: list = None):
        if score_muti is not None:
            score = -val_loss
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model, path)
                self.best_score_muti = score_muti
            elif score < self.best_score + self.delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                for i, sc in enumerate(score_muti):
                    if sc < self.best_score_muti[i] + 0.05:  # self.delta:
                        self.counter += 1
                        print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                        if self.counter >= self.patience:
                            self.early_stop = True
                            break
                        return
                self.save_checkpoint(val_loss, model, path)
                self.best_score = score
                self.best_score_muti = score_muti
                self.counter = 0
        else:
            score = -val_loss
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model, path)
            elif score < self.best_score + self.delta:
                self.counter += 1
                # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model, path)
                self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), os.path.join(path, f'checkpoint_{self.trial_id}.pth'))
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, start_lr, lradj="type1"):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if lradj == 'type1':
        lr_adjust = {epoch: start_lr * (0.5 ** ((epoch - 1) // 1))}
    elif lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # print('Updating learning rate to {}'.format(lr))


def fit(model, optimizer, loss_fn, trainloader, testloader, epoch, device,
        lr_scheduler=None, num_class=2, classification="binary"):
    torch.cuda.empty_cache()
    running_loss = 0
    a = time.time()

    # 模型进入训练状态
    model.train()
    for x, y, padding_mask in trainloader:
        # 根据设备训练，将其切换成所选GPU
        x = x.to(device)
        y = y.reshape(-1).long().to(device)
        padding_mask = padding_mask.to(device)

        # 梯度清零
        optimizer.zero_grad()
        # 模型预测结果
        y_pred = model(x, padding_mask, None, None, None)
        # 计算损失值

        if classification == "TSER":
            y_pred = y_pred.squeeze(dim=1)
            loss = loss_fn(y_pred.float(), y.float())
        else:
            loss = loss_fn(y_pred, y)
        # 损失值反向传播给网络
        loss.backward()
        # 更新网络参数
        optimizer.step()


        running_loss += loss.item()

    if lr_scheduler != None:
        lr_scheduler.step()

    epoch_loss = running_loss / len(trainloader.dataset)
    # test_result = [test_total_T, test_total_TP, test_total_TN, test_total_F, test_total_FN, test_total_FP]
    test_running_loss = 0

    # -------------------------------------------TSER Eval------------------------------------------------------------------
    if classification == "TSER":
        # 模型进入验证状态
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for x, y, padding_mask in testloader:
                # 与训练状态一致
                x = x.to(device)
                y = y.reshape(-1).long().to(device)
                padding_mask = padding_mask.to(device)

                y_pred = model(x, padding_mask, None, None, None)
                loss = loss_fn(y_pred.float(), y.float())

                preds.append(y_pred.detach().cpu())
                trues.append(y)

                test_running_loss += loss.item()

                preds.append(y_pred.detach().cpu())
                trues.append(y)

            trues = torch.cat(trues, 0).detach().cpu().numpy()
            preds = torch.cat(preds, 0).detach().cpu().numpy()

            mae = mean_absolute_error(trues, preds)
            mse = mean_squared_error(trues, preds)
            rmse = sqrt(mse)

        epoch_test_loss = test_running_loss / len(testloader.dataset)

        print_result(epoch, epoch_loss, mae, rmse, a)
        return epoch_loss, epoch_test_loss, mae, rmse



def print_result(epoch, epoch_loss,  mae, rmse,
                a):
    print(
        "epoch:", epoch + 1, "\t",
        "loss: ", round(epoch_loss, 3), "\t",
        "mae: ", round(float(mae), 3), "\t",
        "rmse: ", round(float(rmse), 3), "\t",
        "time:", round(time.time() - a, 2)
    )


