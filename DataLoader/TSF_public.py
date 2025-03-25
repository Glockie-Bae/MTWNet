import time
import optuna
import os
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from utils.data_factory import data_provider
from fit.fit import fit, EarlyStopping, adjust_learning_rate, Take_best_score
from load import UEAloader_xi5
from uea import collate_fn_relation
from utils.metrics import metric




class TSF_Fit():
    def __init__(self, args, Model):
        self.dataset_name_list = [

                args.data_path
            ]
        args.root_path = f"F:\_Sorrow\SCNU_M\研二\TSF\Time-Series-Library-main\dataset"
        self.args = args
        self.dataset_accuary = {}
        self.Model = Model
        self.lr = 0.001
        self.public_data = args.public_data



    def Fit(self):
        for dataset_name in self.dataset_name_list:
            time_start = time.time()
            print(f"--------------------Dataset : {dataset_name} ----------------------------")
            path = self.args.dataset_path
            name = dataset_name
            self.train_dl, self.test_dl = self.data_loader(path, name, public_data=self.args.public_data)
            study = optuna.create_study(direction="maximize")
            study.optimize(self.objective, n_trials=self.args.trial)

            trial = study.best_trial
            print("best rmse", -trial.value)
            print("best trial number", trial.number)

            model = self.Model(self.args).to(self.args.device)
            best_rmse = round(-trial.value, 3)
            best_trail = trial.number

            trial_check_path = os.path.join(self.args.check_path, f"checkpoint_{trial.number}.pth")

            model.load_state_dict(torch.load(trial_check_path))

            mae, mse, rmse = self.Take_best_score(self.Model, trial_check_path, self.test_dl, self.args.device)
            print(f"Accuracy: {mae:.3f} recall: {mse:.3f} f1: {rmse:.3f}]")

            time_end = time.time()
            print("time cost", round(time_end - time_start, 3), "s")

            print(f"Dataset : {dataset_name}, Best MSE : {mse} Best MAE : {mae} Best Trail : {best_trail}")
            self.dataset_accuary[dataset_name] = [mse, mae]

            print("-------------------------------------------")
            for key, value in self.dataset_accuary.items():
                print(f"Dataset : {key}, Best mse : {value[0]}, Best mae : {value[1]}")

            #info_json = json.dumps(self.dataset_accuary, sort_keys=False, indent=4, separators=(',', ': '))
            # 显示数据类型
            with open(f'./result/TSF_public/{self.args.model}_baseline_test.txt', 'a+') as writers:  # 打开文件
                for key, value in self.dataset_accuary.items():
                    writers.write(f"\nDataset : {key}, Best mse : {value[0]}, Best mae : {value[1]}")
            writers.close()

    def objective(self, trial):
        trial_id = trial.number
        self.args.lr = trial.suggest_float("lr", 0.0001, 0.1)

        bare = 0

        res = None
        while res is None:
            if bare > 2:
                print("Model fails to get out the polo.")
                return 0.1
            elif bare > 0:
                print(f"Model falls into polo {bare} times, reset anything.")
            bare += 1
            res = self.train(trial_id)
        return res

    def Take_best_score(self, model, path_checkpoint, testloader, device):
        model = self.Model(self.args).to(self.args.device)
        model.eval()
        with torch.no_grad():
            model.load_state_dict(torch.load(path_checkpoint))
            preds = []
            trues = []
            for x, y, padding_mask in testloader:
                # 与训练状态一致
                x = x.to(device)
                y = y.to(device)
                padding_mask = padding_mask.to(device)

                y_pred = model(x, padding_mask, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = y_pred[:, -self.args.pred_len:, f_dim:]
                batch_y = y[:, -self.args.pred_len:, f_dim:]



                # one_predict = y[0, :, 0].detach().cpu()
                # one_predict[-self.args.pred_len:] = outputs[0, :, 0]
                # one_y = y[0, :, 0].detach().cpu()

                # 绘制预测数据和真实数据

                # plt.plot(one_y, label='Ground Truth', color='#2F7FC1')
                # plt.plot(one_predict, label='Predicted', color='#F3D226')
                # # 添加标题和标签
                # plt.title('Prediction vs Ground Truth')
                # plt.xlabel('Time Step')
                # plt.ylabel('Value')
                #
                # # 添加图例
                # plt.legend()
                #
                # # 显示图像
                # plt.show()

                preds.append(outputs.detach().cpu())
                trues.append(batch_y.detach().cpu())

            # mae = mean_absolute_error(trues, preds)
            # mse = mean_squared_error(trues, preds)
            # rmse = math.sqrt(mse)
            # print("-----------------------")
            # print('oringin mse:{}, mae:{}'.format(mse, mae))

            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)

            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('library mse:{}, mae:{}'.format(mse, mae))
            print("-----------------------")


            return mae, mse, rmse

    def train(self, trial_id):
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>curren trialt: {trial_id}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        model = self.Model(self.args)

        # 交叉熵损失函数
        loss_fn = nn.CrossEntropyLoss()
        if self.args.classification == "TSER" or self.args.task_name == "long_term_forecast":
            loss_fn = nn.MSELoss()
        # Adam优化
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)

        model = model.to(self.args.device)
        early_stop = EarlyStopping(trial_id, patience=50, verbose=False)
        epoches = self.args.epoch_count
        torch.cuda.empty_cache()
        Continue_Training = False
        polo_times = 0

        for epoch in range(epoches):
            # 训练结束，保存模型权重
            if Continue_Training:
                model.load_state_dict(torch.load(os.path.join(self.args.check_path, f"checkpoint_{trial_id}.pth")))

            epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc, train_mse, test_mse = fit(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                trainloader=self.train_dl, testloader=self.test_dl,
                epoch=epoch,
                device=self.args.device,
                lr_scheduler=None,
                num_class=self.args.num_class,
                classification=self.args.classification,
                args = self.args
            )

            early_stop(test_mse, model, self.args.check_path)


            if early_stop.early_stop:
                print("early stop")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(optimizer, epoch + 1, self.args.lr)

            model.load_state_dict(torch.load(os.path.join(self.args.check_path, f"checkpoint_{trial_id}.pth")))

        return early_stop.best_score

    def data_loader(self, path, name, public_data):
        train_data, train_loader = data_provider(self.args, flag='train')
        #vali_data, vali_loader = data_provider(self.args, flag='val')
        test_data, test_loader = data_provider(self.args, flag='test')


        return train_loader, test_loader