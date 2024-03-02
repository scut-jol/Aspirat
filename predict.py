import torch
from AttnModel import AttnSleep
import torchaudio
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from util import padding, frame, Advanced_post_processing, compute_overlap_ratio, frame_cut, get_label
from mean_teacher import AUDIO_SAMPLE_RATE, IMU_SAMPLE_RATE, GAS_SAMPLE_RATE, WIN_DARTION, HOP_DURATION

audio_sample_rate = AUDIO_SAMPLE_RATE
audio_win_sample = int(WIN_DARTION * AUDIO_SAMPLE_RATE)
audio_hop_sample = int(HOP_DURATION * AUDIO_SAMPLE_RATE)
imu_sample_rate = IMU_SAMPLE_RATE
imu_win_sample = int(WIN_DARTION * IMU_SAMPLE_RATE)
imu_hop_sample = int(HOP_DURATION * IMU_SAMPLE_RATE)
gas_sample_rate = GAS_SAMPLE_RATE
gas_win_sample = int(WIN_DARTION * GAS_SAMPLE_RATE)
gas_hop_sample = int(HOP_DURATION * GAS_SAMPLE_RATE)

audio_path = "SegmentSwallow/healthy/20230912/邓祥祥_3_3ml_1_2_7/audio.wav"
imu_path = "SegmentSwallow/healthy/20230912/邓祥祥_3_3ml_1_2_7/imu.csv"
gas_path = "SegmentSwallow/healthy/20230912/邓祥祥_3_3ml_1_2_7/gas.csv"
pth_path = "model/AttnModel.pth"


def prediction(audio_path, imu_path, gas_path):
    audio, sr = torchaudio.load(audio_path)
    imu = pd.read_csv(imu_path)
    gas = pd.read_csv(gas_path)
    imu_tensor = torch.tensor(imu[['X', 'Y', 'Z']].values, dtype=torch.float32).T
    gas_tensor = torch.tensor(gas[['value']].values, dtype=torch.float32).T
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != AUDIO_SAMPLE_RATE:
        resample = torchaudio.transforms.Resample(sr, AUDIO_SAMPLE_RATE)
        audio = resample(audio)
    audio = padding(audio, audio_win_sample, audio_hop_sample)
    imu_tensor = padding(imu_tensor, imu_win_sample, imu_hop_sample)
    gas_tensor = padding(gas_tensor, gas_win_sample, gas_hop_sample)
    audio_frames = frame(audio, audio_win_sample, audio_hop_sample)
    imu_frames = frame(imu_tensor, imu_win_sample, imu_hop_sample)
    gas_frames = frame(gas_tensor, gas_win_sample, gas_hop_sample)
    if not (audio_frames.shape[0] == imu_frames.shape[0] == gas_frames.shape[0]):
        min_length = min(audio_frames.shape[0], imu_frames.shape[0], gas_frames.shape[0])
        audio_frames = audio_frames[:min_length, :, :]
        imu_frames = imu_frames[:min_length, :, :]
        gas_frames = gas_frames[:min_length, :, :]
    with torch.no_grad():
        audio_spec = audio_frames.to(device)
        imu_spec = imu_frames.to(device)
        gas_sqen = gas_frames.to(device)
        output = model(audio_spec, imu_spec, gas_sqen)
    events = Advanced_post_processing(output)
    for event in events:
        print(f"swallow onset={event[0]}, offset={event[1]}")


def calculateMetric(data_path):
    metaData = pd.read_csv(data_path)
    count = 0
    total_accuracy, total_sensitivity, total_specificity, total_roc_auc, total_overlap_ratio = 0, 0, 0, 0, 0
    for index, row in metaData.iterrows():
        file_path = metaData.iloc[index, 0]
        audio_path = f"{file_path}/audio.wav"
        imu_path = f"{file_path}/imu.csv"
        gas_path = f"{file_path}/gas.csv"
        Annotated_path = f"{file_path}/Annotated.json"
        json_data_list = get_label(Annotated_path)
        audio, sr = torchaudio.load(audio_path)
        imu = pd.read_csv(imu_path)
        gas = pd.read_csv(gas_path)
        imu_tensor = torch.tensor(imu[['X', 'Y', 'Z']].values, dtype=torch.float32).T
        gas_tensor = torch.tensor(gas[['value']].values, dtype=torch.float32).T

        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        if sr != AUDIO_SAMPLE_RATE:
            resample = torchaudio.transforms.Resample(sr, AUDIO_SAMPLE_RATE)
            audio = resample(audio)

        audio = padding(audio, audio_win_sample, audio_hop_sample)
        imu_tensor = padding(imu_tensor, imu_win_sample, imu_hop_sample)
        gas_tensor = padding(gas_tensor, gas_win_sample, gas_hop_sample)
        audio_frames = frame(audio, audio_win_sample, audio_hop_sample)
        imu_frames = frame(imu_tensor, imu_win_sample, imu_hop_sample)
        gas_frames = frame(gas_tensor, gas_win_sample, gas_hop_sample)
        labels = frame_cut(len(audio_frames), json_data_list)
        if not (audio_frames.shape[0] == imu_frames.shape[0] == gas_frames.shape[0]):
            min_length = min(audio_frames.shape[0], imu_frames.shape[0], gas_frames.shape[0])
            audio_frames = audio_frames[:min_length, :, :]
            imu_frames = imu_frames[:min_length, :, :]
            gas_frames = gas_frames[:min_length, :, :]
            labels = labels[:min_length]

        with torch.no_grad():
            audio_spec = audio_frames.to(device)
            imu_spec = imu_frames.to(device)
            gas_sqen = gas_frames.to(device)
            y_pred = model(audio_spec, imu_spec, gas_sqen)
            y_pred_cpu = y_pred.cpu()
            binary_predictions = torch.where(y_pred_cpu >= 0.5, torch.tensor(1), torch.tensor(0)).view(-1)
            labels = labels.view(-1)
            accuracy = accuracy_score(binary_predictions, labels)
            TP = torch.sum((binary_predictions == 1) & (labels == 1)).item()
            TN = torch.sum((binary_predictions == 0) & (labels == 0)).item()
            FP = torch.sum((binary_predictions == 1) & (labels == 0)).item()
            FN = torch.sum((binary_predictions == 0) & (labels == 1)).item()
            sensitivity = TP / (TP + FN)
            specificity = TN / (TN + FP)
            roc_auc = roc_auc_score(labels, y_pred_cpu.view(-1))
            events = Advanced_post_processing(y_pred)
            start = json_data_list[0]['strat']
            end = json_data_list[0]['end']
            if len(events) > 0:
                overlap_ratio = compute_overlap_ratio(events[0][0], events[0][1], start, end)
            else:
                overlap_ratio = 0
            total_accuracy += accuracy
            total_sensitivity += sensitivity
            total_specificity += specificity
            total_roc_auc += roc_auc
            total_overlap_ratio += overlap_ratio
            count += 1
            print(f"accuracy={accuracy:.2f} "
                  f"sensitivity={sensitivity:.2f} "
                  f"specificity={specificity:.2f} "
                  f"roc_auc={roc_auc:.2f} "
                  f"overlap_ratio={overlap_ratio:.2f}")
            if overlap_ratio == 0:
                print(f"{audio_path} onset={start:.2f} offset={end:.2f} duartion={end-start:.2f}")
    print(f"Avg_accuracy={(total_accuracy / count):.2f} "
          f"Avg_sensitivity={(total_sensitivity / count):.2f} "
          f"Avg_specificity={(total_specificity / count):.2f} "
          f"Avg_roc_auc={(total_roc_auc / count):.2f} "
          f"Avg_overlap_ratio={(total_overlap_ratio / count):.2f}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda:3"
    else:
        device = "cpu"

    # audio_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    #     sample_rate=AUDIO_SAMPLE_RATE,
    #     n_fft=1024,
    #     hop_length=256,
    #     n_mels=128
    # ).to(device)
    # imu_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    #     sample_rate=IMU_SAMPLE_RATE,
    #     n_fft=256,
    #     hop_length=64,
    #     n_mels=128
    # ).to(device)
    # gas_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    #     sample_rate=GAS_SAMPLE_RATE,
    #     n_fft=128,
    #     hop_length=32,
    #     n_mels=64
    # ).to(device)

    model = AttnSleep().to(device)
    model.load_state_dict(torch.load(pth_path))
    model.eval()

    prediction(audio_path, imu_path, gas_path)
    # calculateMetric("test_data.csv")
