from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from dataset import Aspiration, collate_fn
from torch.utils.data import DataLoader
from swollow_model import *
from cov19model import CoughBreathSpeechModel
from sleep_stage import SleepStageModel
from MultiModel import MultiModel
from train import Trainclass
from torch.optim.lr_scheduler import OneCycleLR
import json
import pandas as pd
import statistics
import os
import pprint
from data_util import predict_calculate_overlap, initialize_weights_uniform, initialize_weights_normal


def main():
    with open('config.json', 'r') as json_file:
        config_dict = json.load(json_file)
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=config_dict['BatchSize'], metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=config_dict['TestBatchSize'], metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=config_dict['Epochs'], metavar='N',
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=config_dict['LearningRate'], metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=config_dict['Momentum'], metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=config_dict['DisableGpu'],
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=config_dict['Seed'], metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=config_dict['LogInterval'], metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--label_meta', type=str, default=config_dict['LabelMeta'], metavar='N',
                        help='data dir path')
    parser.add_argument('--unlabel_meta', type=str, default='unlabel_meta.csv', metavar='N',
                        help='data dir path')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--output-pth', type=str, default=f'{config_dict["Model"]}_{config_dict["Cuda"]}.pth', metavar='N',
                        help='save path of pth file')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{config_dict['Cuda']}" if use_cuda else "cpu")

    kwargs = {'num_workers': config_dict['NumWorkers']} if use_cuda else {}

    data = pd.DataFrame(os.listdir(config_dict['train_dir']))
    kf = KFold(n_splits=config_dict['Splits'], shuffle=True, random_state=args.seed)
    fold_avg_acc, fold_avg_sen, fold_avg_spe, fold_avg_auc, fold_avg_jsc = [], [], [], [], []
    for fold, (train_idx, val_idx) in enumerate(kf.split(data), 1):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[val_idx]
        Aspiration_train_dataset = Aspiration(train_data, config_dict['train_dir'], args.batch_size)
        Aspiration_test_dataset = Aspiration(test_data, config_dict['train_dir'], args.test_batch_size)

        train_loader = DataLoader(
            Aspiration_train_dataset,
            batch_size=1,
            collate_fn=collate_fn,
            **kwargs
        )

        test_loader = DataLoader(
            Aspiration_test_dataset,
            batch_size=1,
            collate_fn=collate_fn,
            **kwargs
        )
        model_class = globals()[config_dict['Model']]
        model = model_class()
        model.apply(initialize_weights_uniform)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
        pprint.pprint(config_dict)
        print(model)
        loss_fn = torch.nn.BCELoss(reduction='none')

        # 设置 Warmup 的 scheduler
        scheduler = OneCycleLR(optimizer,
                               max_lr=args.lr,
                               steps_per_epoch=int(len(train_loader)),
                               epochs=args.epochs,
                               anneal_strategy='cos')
        train = Trainclass(args,
                           model,
                           loss_fn,
                           device,
                           train_loader,
                           test_loader,
                           optimizer,
                           scheduler,
                           config_dict)
        fold_acc, fold_sen, fold_spe, fold_auc = train.run(fold)
        fold_avg_acc.append(fold_acc)
        fold_avg_sen.append(fold_sen)
        fold_avg_spe.append(fold_spe)
        fold_avg_auc.append(fold_auc)
        jsc_list = predict_calculate_overlap(model, config_dict['train_dir'], test_data, args.output_pth, device)
        fold_jsc = statistics.mean(jsc_list)
        fold_avg_jsc.append(fold_jsc)
        print(f"Fold[{fold}] Avg_jsc={fold_jsc:.2f}%(±{statistics.stdev(jsc_list)})")
    print(f"Final Avg Result: avg_acc={statistics.mean(fold_avg_acc):.2f}%(±{statistics.stdev(fold_avg_acc)}) "
          f"avg_sen={statistics.mean(fold_avg_sen):.2f}% (±{statistics.stdev(fold_avg_sen)}) " +
          f"avg_spe={statistics.mean(fold_avg_spe):.2f}% (±{statistics.stdev(fold_avg_spe)}) "
          f"avg_auc={statistics.mean(fold_avg_auc):.2f}% (±{statistics.stdev(fold_avg_auc)}) "
          f"avg_jsc={statistics.mean(fold_avg_jsc):.2f}% (±{statistics.stdev(fold_avg_jsc)})")


if __name__ == '__main__':
    main()
