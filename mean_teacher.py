from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import torchaudio
from sklearn.model_selection import KFold
from dataset import Aspiration, collate_fn
from torch.utils.data import DataLoader
from AttnModel import AttnSleep, AttnSleep2D
from mean_train import Trainclass
import json
import os
import pandas as pd


def main():
    with open('config.json', 'r') as json_file:
        config_dict = json.load(json_file)
    AUDIO_SAMPLE_RATE = config_dict['AudioSampleRate']
    IMU_SAMPLE_RATE = config_dict['ImuSampleRate']
    GAS_SAMPLE_RATE = config_dict['GasSampleRate']
    # WIN_DARTION = config_dict['WinDuration']
    # HOP_DURATION = config_dict['HopDuration']
    # BIN_NUM = config_dict['BinNum']
    # Training settings
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
    parser.add_argument('--output-pth', type=str, default='AttnModel.pth', metavar='N',
                        help='save path of pth file')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{config_dict['Cuda']}" if use_cuda else "cpu")

    kwargs = {'num_workers': config_dict['NumWorkers']} if use_cuda else {}

    audio_mel_spectrogram = torchaudio.transforms.MFCC(
        sample_rate=AUDIO_SAMPLE_RATE,
        n_mfcc=64,
        melkwargs={
            "n_fft": 2048,
            "n_mels": 64,
            "hop_length": 800,
            "mel_scale": "htk",
        },
    )
    imu_mel_spectrogram = torchaudio.transforms.MFCC(
        sample_rate=IMU_SAMPLE_RATE,
        n_mfcc=64,
        melkwargs={
            "n_fft": 256,
            "n_mels": 64,
            "hop_length": 50,
            "mel_scale": "htk",
        },
    )
    gas_mel_spectrogram = torchaudio.transforms.MFCC(
        sample_rate=AUDIO_SAMPLE_RATE,
        n_mfcc=64,
        melkwargs={
            "n_fft": 26,
            "n_mels": 64,
            "hop_length": 5,
            "mel_scale": "htk",
        },
    )

    data = pd.read_csv(f"{config_dict['train_dir']}/meta.csv")
    kf = KFold(n_splits=config_dict['Splits'], shuffle=False, random_state=None)
    # kf = KFold(n_splits=config_dict['Splits'], shuffle=True, random_state=args.seed)
    fold_avg_acc, fold_avg_sen, fold_avg_spe, fold_avg_auc = 0, 0, 0, 0
    for fold, (train_idx, val_idx) in enumerate(kf.split(data), 1):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[val_idx]
        Aspiration_train_dataset = Aspiration(train_data, config_dict['train_dir'])
        Aspiration_test_dataset = Aspiration(test_data, config_dict['train_dir'])

        # unlabel_data = pd.read_csv(args.unlabel_meta)
        # Aspiration_unlabel_dataset = Aspiration(unlabel_data,
        #                                         bins_num=BIN_NUM,
        #                                         win_duration=WIN_DARTION,
        #                                         hop_duartion=HOP_DURATION,
        #                                         label=False)

        # combined_dataset = torch.utils.data.ConcatDataset([Aspiration_train_dataset, Aspiration_unlabel_dataset])
        train_loader = DataLoader(
            Aspiration_train_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            **kwargs
        )

        test_loader = DataLoader(
            Aspiration_test_dataset,
            batch_size=args.test_batch_size,
            collate_fn=collate_fn,
            **kwargs
        )
        model_class = globals()[config_dict['Model']]
        model = model_class()
        # mean_teacher = AttnSleep()
        model = model.to(device)
        # mean_teacher = mean_teacher.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = torch.nn.BCELoss(reduction='none')
        # loss_fn = torch.nn.BCELoss(reduction='none')
        # loss_cons_fn = torch.nn.MSELoss()
        # scheduler = OneCycleLR(optimizer,
        #                        max_lr=args.lr,
        #                        steps_per_epoch=int(len(train_loader)),
        #                        epochs=args.batch_size,
        #                        anneal_strategy='cos')
        train = Trainclass(args,
                           model,
                           None,
                           loss_fn,
                           None,
                           device,
                           train_loader,
                           test_loader,
                           optimizer,
                           audio_mel_spectrogram,
                           imu_mel_spectrogram,
                           gas_mel_spectrogram,
                           None)
        fold_acc, fold_sen, fold_spe, fold_auc = train.run(fold)
        fold_avg_acc += fold_acc
        fold_avg_sen += fold_sen
        fold_avg_spe += fold_spe
        fold_avg_auc += fold_auc
    print(f"Final Avg Result: avg_acc={fold_avg_acc/10 :2f}% avg_sen={fold_avg_sen/10 :2f}% " +
          f"avg_spe={fold_avg_spe/10 :2f}% avg_auc={fold_avg_auc/10 :2f}")


if __name__ == '__main__':
    main()
