import time
import optuna
import os
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
import json
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append("..")
from fit.fit import fit, EarlyStopping, adjust_learning_rate, Take_best_score
from load import UEAloader_xi5
from uea import collate_fn_relation




class TSC_Fit():
    def __init__(self, args, Model):
        self.dataset_name_list = [

                "ArticularyWordRecognition",
                "AtrialFibrillation", "BasicMotions",
                "CharacterTrajectories",
                "Cricket",
                "DuckDuckGeese", "Epilepsy", "ERing",
                # "EigenWorms",
                "EthanolConcentration", "FaceDetection", "FingerMovements", "HandMovementDirection",
                "Handwriting", "Heartbeat", "InsectWingbeat",
                "JapaneseVowels", "Libras", "LSST",
                "MotorImagery", "NATOPS", "PenDigits",
                "PEMS-SF", "PhonemeSpectra", "RacketSports", "SelfRegulationSCP1", "SelfRegulationSCP2",
                "StandWalkJump", "UWaveGestureLibrary", "SpokenArabicDigits",
            ]
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
            print("best acc", trial.value)
            print("best trial number", trial.number)

            model = self.Model(self.args).to(self.args.device)
            best_acc = trial.value
            best_trail = trial.number

            trial_check_path = os.path.join(self.args.check_path, f"checkpoint_{trial.number}.pth")

            model.load_state_dict(torch.load(trial_check_path))

            accuracy, recall, f1 = self.Take_best_score(self.Model, trial_check_path, self.test_dl, self.args.device)
            print(f"Accuracy: {accuracy:.3f} recall: {recall:.3f} f1: {f1:.3f}]")

            time_end = time.time()
            print("time cost", time_end - time_start, "s")

            print(f"Dataset : {dataset_name}, Best Accuracy : {best_acc} Best Trail : {best_trail}")
            self.dataset_accuary[dataset_name] = best_acc

            print("-------------------------------------------")
            for key, value in self.dataset_accuary.items():
                print(f"Dataset : {key}, Best Accuracy : {value}")

            info_json = json.dumps(self.dataset_accuary, sort_keys=False, indent=4, separators=(',', ': '))
            # 显示数据类型
            f = open(f'./result/TSC/{self.args.model}_test.json', 'w')
            f.write(info_json)
            f.close()

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
                x = x.to(device)
                padding_mask = padding_mask.to(device)
                # corr      = corr.to(device)

                y_pred = model(x, padding_mask, None, None, None)
                y_pred = torch.argmax(y_pred, dim=1)

                preds.append(y_pred.detach().cpu())
                trues.append(y)

            trues = torch.cat(trues, 0).detach().cpu()
            preds = torch.cat(preds, 0).detach().cpu()

            accuracy = accuracy_score(trues, preds)
            recall = recall_score(trues, preds, average='weighted')
            f1 = f1_score(trues, preds, average='weighted')

            return accuracy, recall, f1

    def train(self, trial_id):
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>curren trialt: {trial_id}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        model = self.Model(self.args)

        # 交叉熵损失函数
        loss_fn = nn.CrossEntropyLoss()
        if self.args.classification == "TSER":
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

            epoch_loss, epoch_acc, epoch_test_loss, epoch_test_acc, test_recall, test_f1 = fit(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                trainloader=self.train_dl, testloader=self.test_dl,
                epoch=epoch,
                device=self.args.device,
                lr_scheduler=None,
                num_class=self.args.num_class,
                classification=self.args.classification,
                args=self.args
            )

            early_stop(-epoch_test_acc, model, self.args.check_path)
            if early_stop.early_stop:
                print("early stop")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(optimizer, epoch + 1, self.args.lr)
            # 这种训练下去几乎毫无意义
            if test_recall < 0.10:
                if polo_times >= 50:
                    return None
                polo_times += 1
                # print(f"polo count {polo_times}")
            else:
                polo_times = 0
            model.load_state_dict(torch.load(os.path.join(self.args.check_path, f"checkpoint_{trial_id}.pth")))

        return early_stop.best_score

    def data_loader(self, path, name, public_data):
        train_ds = UEAloader_xi5(
            dataset_path=path,
            dataset_name=name,
            flag="train",
            public_data=public_data
        )

        test_ds = UEAloader_xi5(
            dataset_path=path,
            dataset_name=name,
            flag="test",
            public_data=public_data
        )

        self.args.channels = 3

        # args.batch_size = 6
        batch_size = self.args.batch_size
        num_workers = 0  # win 平台下是0，否则出问题；linux可以设置12

        seq_len = test_ds.data.shape[1]
        self.args.seq_len = seq_len

        self.args.num_class = len(np.unique(test_ds.labels))

        # args.num_class = 2 # 记得改output，MergeModel输出为2
        self.args.enc_in = test_ds.data.shape[2]
        self.args.dec_in = test_ds.data.shape[2]
        # args.enc_in = test_ds.data.shape[2]

        # args.p_hidden_dims = [2, 2]
        # args.p_hidden_layers = 2
        self.args.router_k = self.args.enc_in
        self.args.label_len = 2
        self.args.pred_len = 0

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=lambda x: collate_fn_relation(x, max_len=seq_len)
        )

        test_dl = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=lambda x: collate_fn_relation(x, max_len=seq_len)
        )

        return train_dl, test_dl