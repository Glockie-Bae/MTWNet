import os
import time
import numpy as np
import matplotlib.pyplot as plt

from thop import profile
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import torch.nn.functional as F
import tqdm
import sys
sys.path.append("..")
from utils.metrics import metric
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

    def __call__(self, val_loss, model, path, score_muti:list=None):
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
                    if sc < self.best_score_muti[i] + 0.05: # self.delta:
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
                #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
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
        #print('Updating learning rate to {}'.format(lr))

def fit(model, optimizer, loss_fn, trainloader, testloader, epoch, device, args,
lr_scheduler=None, num_class=2, classification = "binary"):
    torch.cuda.empty_cache()
    correct = [0]
    total   = 0
    running_loss = 0
    a = time.time()

    # 模型进入训练状态
    model.train()
    for x, y, padding_mask in trainloader:
        # 根据设备训练，将其切换成所选GPU

        if classification == "long_term_forecast":
            x = x.to(device)
            y = y.to(device)
        else:
            x = x.to(device)
            y = y.reshape(-1).long().to(device)
        padding_mask    = padding_mask.to(device)

        # 梯度清零
        optimizer.zero_grad()
        # 模型预测结果

        # flops, params = profile(model, inputs=(x, padding_mask, None, None, None))
        # print("--------------------------------")
        # print(flops, params)
        # print("--------------------------------")
        # assert True, "该代码只能在 Linux 下执行"
        y_pred = model(x, padding_mask, None, None, None)
        # 计算损失值
        if classification == "TSER":
            y_pred = y_pred.squeeze(dim=1)
            loss = loss_fn(y_pred.float(), y.float())
        elif classification == "long_term_forecast":
            f_dim = -1 if args.features == 'MS' else 0
            outputs = y_pred[:, -args.pred_len:, f_dim:]
            batch_y = y[:, -args.pred_len:, f_dim:]
            loss = loss_fn(outputs, batch_y)
        else:
            loss = loss_fn(y_pred, y)
        # 损失值反向传播给网络
        loss.backward()
        # 更新网络参数
        optimizer.step()

        with torch.no_grad():
            if classification == "TSER":
                mse = cal_train_acc_tser(y_pred, y, correct, threshold=7500)
            elif classification == "long_term_forecast":
                train_mse = ((outputs - batch_y) ** 2).mean()
                correct[0] = 0
            else:
                cal_train_acc(y_pred, y, correct)

            # 总共多少行
            total += y.size(0)
            # 计算总损失值
            running_loss += loss.item()
            
    if lr_scheduler != None:
        lr_scheduler.step()

    epoch_loss = running_loss / len(trainloader.dataset)
    train_acc = correct[0] / total
    #test_result = [test_total_T, test_total_TP, test_total_TN, test_total_F, test_total_FN, test_total_FP]
    test_result = [0] * 6
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


                for index, item in enumerate(y):
                    cal_test_acc_tser(y_pred, item, index, test_result)

                test_running_loss += loss.item()


                preds.append(y_pred.detach().cpu())
                trues.append(y)

            trues = torch.cat(trues, 0).detach().cpu().numpy()
            preds = torch.cat(preds, 0).detach().cpu().numpy()

            mae = mean_absolute_error(trues, preds)
            mse = mean_squared_error(trues, preds)
            rmse = sqrt(mse)

        epoch_test_loss = test_running_loss / len(testloader.dataset)
        epoch_test_acc = (test_result[1] + test_result[2] )/ (test_result[0] + test_result[3])
        epoch_test_T_acc = test_result[1] / test_result[0]
        epoch_test_F_acc = test_result[2] / test_result[3]
        print_result(epoch, epoch_loss, epoch_test_acc, epoch_test_T_acc, epoch_test_F_acc, test_result, mae, rmse, train_acc, a)
        return epoch_loss, train_acc, epoch_test_loss, epoch_test_acc, mae, rmse
# ---------------------------------------------Long_term_forecast------------------------------------------------
    elif classification == 'long_term_forecast':
        # 模型进入验证状态
        model.eval()
        preds = []
        trues = []
        #test_running_loss = []
        with torch.no_grad():
            for x, y, padding_mask in testloader:
                # 与训练状态一致
                x = x.to(device)
                y = y.to(device)
                padding_mask = padding_mask.to(device)

                y_pred = model(x, padding_mask, None, None, None)

                f_dim = -1 if args.features == 'MS' else 0
                outputs = y_pred[:, -args.pred_len:, f_dim:]
                batch_y = y[:, -args.pred_len:, f_dim:]
                loss = loss_fn(outputs, batch_y)
                #test_running_loss.append(loss)
                preds.append(outputs.detach().cpu())
                trues.append(batch_y.detach().cpu())

            #test_running_loss = np.array(test_running_loss)
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f"train_mse : {train_mse}, test_mse : {mse}, test_mae : {mae}")
        return epoch_loss, train_acc, 0, 0, train_mse, mse
# -------------------------------------------TSC Eval------------------------------------------------------------------
    else:
        recall_list = []
        f1_list = []
        precision_list = []
        true_dict = dict()
        total_dict = dict()
        true_num = 0
        total_num = 0
        for i in range(num_class):
            true_dict[i] = 0
            total_dict[i] = 0
        # 模型进入验证状态
        model.eval()
        with torch.no_grad():
            for x, y, padding_mask in testloader:
                # 与训练状态一致
                x = x.to(device)
                y = y.reshape(-1).long().to(device)
                padding_mask = padding_mask.to(device)

                y_pred = model(x, padding_mask, None, None, None)

                loss = loss_fn(y_pred, y)
                y_pred = torch.argmax(y_pred, dim=1)

                if num_class == 2:
                    for index, item in enumerate(y):
                        cal_test_acc(item, index, y_pred, test_result)
                else:
                    trues = y.detach().cpu()
                    preds = y_pred.detach().cpu()

                    f1_list.append(f1_score(trues, preds, average='macro'))
                    precision_list.append(precision_score(trues, preds, average='macro'))
                    recall_list.append(recall_score(trues, preds, average='macro'))

                    for i, item in enumerate(trues):
                        total_dict[int(item)] += 1
                        total_num += 1
                        if item == preds[i]:
                            true_dict[int(item)] += 1
                            true_num += 1

                test_running_loss += loss.item()

        if num_class == 2:
            epoch_test_loss = test_running_loss / len(testloader.dataset)
            epoch_test_acc = (test_result[1] + test_result[2]) / (test_result[0] + test_result[3])
            epoch_test_T_acc = test_result[1] / test_result[0]
            epoch_test_F_acc = test_result[2] / test_result[3]

            if (test_result[1] + test_result[5]) == 0:
                precision = 0
                recall = 0
                f1 = 0
            else:
                precision = (test_result[1]) / (test_result[1] + test_result[5])
                recall = (test_result[1]) / (test_result[1] + test_result[4])
                if (precision + recall) == 0.0:
                    f1 = 0.0
                else:
                    f1 = 2 * (precision * recall) / (precision + recall)

            print_result_binrary(epoch, epoch_loss, epoch_test_acc, epoch_test_T_acc, epoch_test_F_acc, test_result, train_acc, a)
        else:
            epoch_test_loss = test_running_loss / len(testloader.dataset)
            epoch_test_acc = true_num / total_num
            precision = np.mean(precision_list)
            recall = np.mean(recall_list)
            f1 = np.mean(f1_list)
            if epoch % 10 == 0:
                print(
                    "epoch:", epoch + 1, "\t",
                    "loss: ", round(epoch_loss, 3), "\t",
                    "acc: ", round(train_acc, 3), "\t",
                    "test_acc: ", round(epoch_test_acc, 3), "\t",
                )

                print(
                    "precision", round(precision, 3), "\t",
                    "recall", round(recall, 3), "\t",
                    "f1", round(f1, 3), "\t",
                    "time:", round(time.time() - a, 2)
                )

        return epoch_loss, train_acc, epoch_test_loss, epoch_test_acc, recall, f1



def Take_best_score(model, path_checkpoint, testloader, device):
    model.eval()
    with torch.no_grad():
        model.load_state_dict(torch.load(path_checkpoint))
        preds = []
        trues = []

        for x, y, padding_mask in testloader:
            x      = x.to(device)
            padding_mask      = padding_mask.to(device)
            # corr      = corr.to(device)

            y_pred = model(x, padding_mask, None, None, None)
            y_pred = torch.argmax(y_pred,dim=1)

            preds.append(y_pred.detach().cpu())
            trues.append(y)

        trues = torch.cat(trues, 0).detach().cpu()
        preds = torch.cat(preds, 0).detach().cpu()

        accuracy = accuracy_score(trues, preds)
        recall = recall_score(trues, preds, average='weighted')
        f1 = f1_score(trues, preds, average='weighted')

        return accuracy, recall, f1

def cal_train_acc_tser(y_pred, y, correct, threshold=7500):
    mse = ((y_pred - y) ** 2).mean()
    for i, item in enumerate(y):
        # label true
        if item >= 7500:
            # label true pred true
            if y_pred[i] >= threshold:
                correct[0] += 1
        elif item < 7500:
            # label true pred true
            if y_pred[i] < threshold:
                correct[0] += 1
    return mse


def cal_train_acc(y_pred, y, correct):
    # 返回每一行中，tensor的最大值
    y_pred = torch.argmax(y_pred, dim=1)
    # 计算预测正确的数量
    correct[0] += (y_pred == y).sum().item()

def cal_test_acc_tser(y_pred, item, index, test_result):
    #test_result = [test_total_T, test_total_TP, test_total_TN, test_total_F, test_total_FN, test_total_FP]
    # label true
    if item >= 7500:
        test_result[0] += 1
        # label true pred true
        if y_pred[index] >= 7500:
            test_result[1] += 1
        # label true pred false
        else:
            test_result[4] += 1
    elif item < 7500:
        test_result[3] += 1
        # label true pred true
        if y_pred[index] < 7500:
            test_result[2] += 1
        # label true pred false
        else:
            test_result[5] += 1

def cal_test_acc(item, i, y_pred, test_result):
    # label true
    if item == 1:
        test_result[0] += 1
        # label true pred true
        if y_pred[i] == 1:
            test_result[1] += 1
        # label true pred false
        else:
            test_result[4] += 1

    # label false
    if item == 0:
        test_result[3] += 1
        # label false pred false
        if y_pred[i] == 0:
            test_result[2] += 1
        # label false pred true
        else:
            test_result[5] += 1

def print_result(epoch, epoch_loss, epoch_test_acc, epoch_test_T_acc, epoch_test_F_acc, test_result, mae, rmse, train_acc, a):
    print(
        "epoch:", epoch + 1, "\t",
        "loss: ", round(epoch_loss, 3), "\t",
        "mae: ", round(float(mae), 3), "\t",
        "rmse: ", round(float(rmse), 3), "\t",
        "train_acc: ", round(train_acc, 3), "\t",
        "test_acc: ", round(epoch_test_acc, 3), "\t",
        "T_Total: ", round(test_result[0], 3), "\t",
        "F_Total: ", round(test_result[3], 3), "\t",
        "TP_Total: ", round(test_result[1], 3), "\t",
        "TN_Total: ", round(test_result[2], 3), "\t",
        "TT_acc: ", round(epoch_test_T_acc, 3), "\t",
        "FT_acc: ", round(epoch_test_F_acc, 3), "\t",
        "time:", round(time.time() - a, 2)
    )

def print_result_binrary(epoch, epoch_loss, epoch_test_acc, epoch_test_T_acc, epoch_test_F_acc, test_result, train_acc, a):
    print(
        "epoch:", epoch + 1, "\t",
        "loss: ", round(epoch_loss, 3), "\t",
        "train_acc: ", round(train_acc, 3), "\t",
        "test_acc: ", round(epoch_test_acc, 3), "\t",
        "T_Total: ", round(test_result[0], 3), "\t",
        "F_Total: ", round(test_result[3], 3), "\t",
        "TP_Total: ", round(test_result[1], 3), "\t",
        "TN_Total: ", round(test_result[2], 3), "\t",
        "TT_acc: ", round(epoch_test_T_acc, 3), "\t",
        "FT_acc: ", round(epoch_test_F_acc, 3), "\t",
        "time:", round(time.time() - a, 2)
    )