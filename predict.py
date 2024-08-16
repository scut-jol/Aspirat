import torch
from swollow_model import CFSCNet
import torchaudio
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
import os
from data_util import mix, resample, padding, frame, post_process, calculate_jaccard_index
import json


def prediction(model, dir_path, device):
    with open('config.json', 'r') as json_file:
        config_dict = json.load(json_file)
    win_duration = config_dict['WinDuration']
    hop_duartion = config_dict['HopDuration']
    audio_sample_rate = config_dict['AudioSampleRate']
    audio_win_sample = int(win_duration * audio_sample_rate)
    audio_hop_sample = int(hop_duartion * audio_sample_rate)
    imu_sample_rate = config_dict['ImuSampleRate']
    imu_win_sample = int(win_duration * imu_sample_rate)
    imu_hop_sample = int(hop_duartion * imu_sample_rate)
    gas_sample_rate = config_dict['GasSampleRate']
    gas_win_sample = int(win_duration * gas_sample_rate)
    gas_hop_sample = int(hop_duartion * gas_sample_rate)

    audio_path = os.path.join(dir_path, "audio.wav")
    imu_path = os.path.join(dir_path, "imu.csv")
    gas_path = os.path.join(dir_path, "gas.csv")
    Annotated_path = os.path.join(dir_path, "Annotated.json")
    audio, sr = torchaudio.load(audio_path)
    imu = pd.read_csv(imu_path)
    gas = pd.read_csv(gas_path)
    imu_tensor = torch.tensor(imu[['X', 'Y', 'Z']].values, dtype=torch.float32).T
    gas_tensor = torch.tensor(gas[['value']].values, dtype=torch.float32).T
    audio = mix(audio)
    audio = resample(audio, sr, audio_sample_rate)
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
    assert not torch.isnan(audio_frames).any() and not torch.isnan(imu_frames).any() and not torch.isnan(gas_frames).any()

    model.eval()
    with torch.no_grad():
        audio_frames = audio_frames.to(device)
        imu_frames = imu_frames.to(device)
        gas_frames = gas_frames.to(device)
        output = model(audio_frames, imu_frames, gas_frames)
        sequence = post_process(output, False)
        # torch.save(torch.tensor(sequence), 'sequence.pt')
        events = sequence_to_intervals(sequence)
    with open(Annotated_path, 'r') as json_file:
        label_list = json.load(json_file)['label_list']
    for start, end in events:
        label_start = label_list[0]["start"] * 1000
        label_end = label_list[0]["end"] * 1000
        jsc_ratio = calculate_jaccard_index(start, end, label_start, label_end)
        print(dir_path)
        print(f"swallow onset={start}, offset={end}, label onset={label_start}, offset={label_end}, jsc_ratio={jsc_ratio}")


def sequence_to_intervals(sequence):
    intervals = []
    start = None
    duration = 200  # 每个数值的间隔是200ms

    for index, value in enumerate(sequence):
        if value == 1 and start is None:
            start = index * duration
        elif value == 0 and start is not None:
            end = index * duration
            intervals.append((start, end))
            start = None

    # 如果序列在最后一个位置结束是1，需要处理最后的区间
    if start is not None:
        end = len(sequence) * duration
        intervals.append((start, end))

    return intervals


def predict_calculate_overlap(model, root_path, data_list, pth_path, audio_transfromation, imu_transfromation, gas_transfromation):
    model.load_state_dict(torch.load(pth_path))
    model.eval()
    for index in range(len(data_list.iloc[:, 0])):
        file_path = os.path.join(root_path, data_list.iloc[index, 0])
        loaded_dict = torch.load(file_path)
        audio = loaded_dict['audio']
        imu = loaded_dict['imu']
        gas = loaded_dict['gas']
        label = loaded_dict['labels']
        label = label.to(device)
        audio = audio.to(device)
        imu = imu.to(device)
        gas = gas.to(device)
        audio = audio_transfromation(audio)
        imu = imu_transfromation(imu)
        gas = gas_transfromation(gas)
        output = model(audio, imu, gas)
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
    pth_path = "CFSCNet_0.pth"
    model = CFSCNet().to(device)
    model.load_state_dict(torch.load(pth_path))

    # root_path = "../SegmentSwallow/patient/20230828"
    # for root, dirs, files in os.walk(root_path):
    #     for name in files:
    #         if "Annotated" in name:
    #             prediction(model, root, device)
    #             break
    root_path = "../SegmentSwallow/patient/20231225/郑福永_0_10ml_7_2_11"
    prediction(model, root_path, device)
