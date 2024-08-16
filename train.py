import torch
import torch.nn
from torchaudio.transforms import FrequencyMasking, TimeMasking
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from data_util import post_process, get_weight
import pickle
import numpy as np


class Trainclass():
    def __init__(self,
                 args,
                 model,
                 loss_fn,
                 device,
                 train_loader,
                 test_loader,
                 optimizer,
                 scheduler,
                 config_dict):
        self.args = args
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.masking = [FrequencyMasking(freq_mask_param=10),
                        TimeMasking(time_mask_param=10)]
        self.metric = -150
        self.scheduler = scheduler
        self.config_dict = config_dict
        self.type = config_dict["type"]
        # self.start_weight = False

    def run(self, fold_idx):
        best_acc, best_sen, best_spe, best_auc, best_metric = 0, 0, 0, 0, -100
        early_stop, pre_test_loss = 0, 100
        for i in range(1, self.args.epochs + 1):
            train_loss = self.train()
            test_loss, avg_acc, avg_sen, avg_spe, avg_auc, metric = self.test()
            if metric > best_metric:
                best_auc = avg_auc
                best_acc = avg_acc
                best_sen = avg_sen
                best_spe = avg_spe
                best_metric = metric
            print(f'Fold[{fold_idx}] Epoch: {i} [{i}/{self.args.epochs} ({100. * i / self.args.epochs:.0f}%)] '
                  f'Train loss={train_loss:.6f} Test loss={test_loss:.6f} Current lr={self.scheduler.get_last_lr()}\n')
            # if test_loss < pre_test_loss:
            #     early_stop = 0
            # else:
            #     early_stop += 1
            # pre_test_loss = test_loss
            # if early_stop == 20:
            #     print("Early Stopping!!!")
            #     break
            # if i > 30:
            #     self.start_weight = True
        print(f"Fold[{fold_idx}] Result: acc={best_acc:.2f}% sen={best_sen:.2f}%, spe={best_spe:.2f}%, auc={best_auc:.2f}%!")
        return best_acc, best_sen, best_spe, best_auc

    def train(self):
        self.model.train()
        total_loss = []
        for count, (audio, imu, gas, targets) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            targets = targets.to(self.device)
            audio = audio.to(self.device)
            imu = imu.to(self.device)
            gas = gas.to(self.device)
            output = self.model(audio, imu, gas)
            targets_loss = self.loss_fn(output, targets)
            if self.type == 1 or self.type == 3:
                targets_weight = get_weight(targets)
            else:
                targets_weight = torch.ones_like(targets)
            targets_weight = targets_weight.to(self.device)
            loss = torch.mean(targets_weight * targets_loss)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss.append(loss.item())

            if count % self.args.log_interval == 0:
                print(f"[{count}/{len(self.train_loader)}] Train loss={loss.item()}")
        avg_loss = sum(total_loss) / len(total_loss)
        return avg_loss

    def test(self):
        self.model.eval()
        loss_list = []
        output_list, pred_list, label_list = [], [], []
        with torch.no_grad():
            for (audio, imu, gas, targets) in self.test_loader:
                targets = targets.to(self.device)
                audio = audio.to(self.device)
                imu = imu.to(self.device)
                gas = gas.to(self.device)
                output = self.model(audio, imu, gas)
                if self.type == 2 or self.type == 3:
                    pred = post_process(output)
                else:
                    pred = (output > 0.5).float()
                loss = torch.mean(self.loss_fn(output, targets)).item()

                # 将所有批次的结果展平并添加到列表中
                output_list.extend(output.cpu().numpy().reshape(-1))
                pred_list.extend(pred.cpu().numpy().reshape(-1))
                label_list.extend(targets.cpu().numpy().reshape(-1))

                loss_list.append(loss)

            # 将列表转换为NumPy数组
            output_array = np.array(output_list)
            pred_array = np.array(pred_list)
            label_array = np.array(label_list)
            # 计算整个数据集的AUC
            avg_auc = roc_auc_score(label_array, output_array) * 100

            # 计算整个数据集的准确率
            avg_accuracy = accuracy_score(label_array, pred_array) * 100

            # 计算整个数据集的混淆矩阵
            cm = confusion_matrix(label_array, pred_array, labels=[0, 1])
            TN, FP, FN, TP = cm.ravel()

            # 计算整个数据集的灵敏度和特异性
            avg_sensitivity = (TP / (TP + FN) if (TP + FN) != 0 else 0.0) * 100
            avg_specificity = (TN / (TN + FP) if (TN + FP) != 0 else 0.0) * 100

            # 计算平均损失
            print(f'Test set avg_accuracy={avg_accuracy:.2f}% '
                  f'avg_sensitivity={avg_sensitivity:.2f}%, '
                  f'avg_specificity={avg_specificity:.2f}% '
                  f'avg_auc={avg_auc:.2f}%')
            # metric = (avg_auc - 90) + (avg_accuracy - 83) + (avg_sensitivity - 63) + (avg_specificity - 90)
            metric = avg_auc
            if metric > self.metric:
                torch.save(self.model.state_dict(), self.args.output_pth)
                self.metric = metric
                if self.type == 0:
                    with open(f'{self.config_dict["Model"]}_metrics_data.pkl', 'wb') as file:
                        pickle.dump((output_list, pred_list, label_list), file)
                # with open('metrics_data.pkl', 'rb') as file:
                #     output_list, pred_list, label_list = pickle.load(file)
                print(f"Best model saved!! Metric={self.metric}!!")

            return (sum(loss_list) / len(loss_list), avg_accuracy, avg_sensitivity, avg_specificity, avg_auc, self.metric)
