import torch
import torch.nn
from torchaudio.transforms import FrequencyMasking, TimeMasking
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from data_util import post_process

class Trainclass():
    def __init__(self,
                 args,
                 model,
                 mean_teacher,
                 loss_fn,
                 loss_cons_fn,
                 device,
                 train_loader,
                 test_loader,
                 optimizer,
                 audio_transfromation,
                 imu_transfromation,
                 gas_transfromation,
                 scheduler):
        self.args = args
        self.model = model
        self.mean_teacher = mean_teacher
        self.loss_fn = loss_fn
        self.loss_cons_fn = loss_cons_fn
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.audio_transfromation = audio_transfromation.to(device)
        self.imu_transfromation = imu_transfromation.to(device)
        self.gas_transfromation = gas_transfromation.to(device)
        self.masking = [FrequencyMasking(freq_mask_param=10),
                        TimeMasking(time_mask_param=10)]
        self.avg_auc = 0
        self.scheduler = scheduler

    def run(self, fold_idx):
        best_acc, best_sen, best_spe, best_auc = 0, 0, 0, 0
        for i in range(1, self.args.epochs + 1):
            train_loss = self.train()
            test_loss, avg_acc, avg_sen, avg_spe, avg_auc = self.test()
            if avg_auc > best_auc:
                best_auc = avg_auc
                best_acc = avg_acc
                best_sen = avg_sen
                best_spe = avg_spe
            # tea_loss = self.test(self.mean_teacher)
            print(f'Fold[{fold_idx}] Epoch: {i} [{i}/{self.args.epochs} ({100. * i / self.args.epochs:.0f}%)] '
                  f'Train loss={train_loss:.6f} Test loss={test_loss:.6f}\n')
        print(f"Fold[{fold_idx}] Best Result: acc={best_acc} sen={best_sen}, spe={best_spe}, auc={best_auc}!")
        return best_acc, best_sen, best_spe, best_auc

    def train(self):
        self.model.train()
        total_loss = []
        for count, (audio, imu, gas, targets, targets_weight) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            targets = targets.to(self.device)
            targets_weight = targets_weight.to(self.device)
            output = self.model(audio.to(self.device), imu.to(self.device), gas.to(self.device))
            # forward pass with mean teacher
            # with torch.no_grad():
            #     mean_t_output = self.mean_teacher(audio_masking, imu_masking, gas_sqen)

            # consistency loss
            # const_loss = self.loss_cons_fn(output, mean_t_output)
            # set the consistency weight
            # target_value = 4 / 3
            # targets_weight[targets_weight != target_value] = 4
            targets_loss = self.loss_fn(output, targets)
            loss = torch.mean(targets_weight * targets_loss)
            # loss = self.loss_fn(output, targets)
            loss.backward()
            self.optimizer.step()
            # post_process
            total_loss.append(loss.item())
            # update mean teacher, (should choose alpha somehow)
            # Use the true average until the exponential average is more correct
            # alpha = 0.9
            # for mean_param, param in zip(self.mean_teacher.parameters(), self.model.parameters()):
            #     mean_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
            if count % self.args.log_interval == 0:
                print(f"[{count}/{len(self.train_loader)}] Train loss={loss.item()}")
        avg_loss = sum(total_loss) / len(total_loss)
        return avg_loss

    def test(self):
        self.model.eval()
        accuracy_list, sensitivity_list, specificity_list, auc_list, loss_list = [], [], [], [], []
        with torch.no_grad():
            for (audio, imu, gas, targets, _) in self.test_loader:
                targets = targets.to(self.device)
                output = self.model(audio.to(self.device), imu.to(self.device), gas.to(self.device))
                pred = post_process(output)
                loss = torch.mean(self.loss_fn(output, targets)).item()
                output_flattened = output.cpu().numpy().reshape(-1)
                targets_flattened = targets.cpu().numpy().reshape(-1)
                pred_flattend = pred.cpu().numpy().reshape(-1)
                accuracy = accuracy_score(targets_flattened, pred_flattend)
                TN, FP, FN, TP = confusion_matrix(targets_flattened, pred_flattend).ravel()
                sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0.0
                specificity = TN / (TN + FP) if (TN + FP) != 0 else 0.0
                auc = roc_auc_score(targets_flattened, output_flattened)
                loss_list.append(loss)
                auc_list.append(auc)
                accuracy_list.append(accuracy)
                sensitivity_list.append(sensitivity)
                specificity_list.append(specificity)
            avg_accuracy = sum(accuracy_list) / len(accuracy_list) * 100.
            avg_sensitivity = sum(sensitivity_list) / len(sensitivity_list) * 100.
            avg_specificity = sum(specificity_list) / len(specificity_list) * 100.
            avg_auc = sum(auc_list) / len(auc_list)
            print(f'Test set avg_accuracy={avg_accuracy:.2f}% '
                  f'avg_sensitivity={avg_sensitivity:.2f}%, '
                  f'avg_specificity={avg_specificity:.2f}% '
                  f'avg_auc={avg_auc:.4f}')
            if avg_auc > self.avg_auc:
                torch.save(self.model.state_dict(), self.args.output_pth)
                self.avg_auc = avg_auc
                print("Best model saved!!!!")

            return (sum(loss_list) / len(loss_list), avg_accuracy, avg_sensitivity, avg_specificity, avg_auc)