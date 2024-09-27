import os
import pandas as pd
from scipy import signal
import csv
from sklearn.model_selection import train_test_split
import json
import torchaudio
import torch
import torch.nn as nn
import numpy as np
import time
import librosa
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
from scipy.signal import welch
import torch.nn.init as init


def get_weight(label):
    num_ones = torch.sum(label == 1).item()
    num_zeros = torch.sum(label == 0).item()
    alpha = num_ones / (num_ones + num_zeros)
    weight = label / alpha + (1 - label) / (1 - alpha)
    return weight


# def get_weight(label, t, T):
#     L = 1.5
#     k = 0.1
#     ratio = L / (1 + np.exp(-k * (t - (T // 4))))
#     num_ones = torch.sum(label == 1).item()
#     num_zeros = torch.sum(label == 0).item()
#     num_sum = num_ones + num_zeros
#     q_zero = num_sum / num_zeros
#     q_one = num_sum / num_ones
#     weight = torch.ones_like(label)
#     weight[label == 1] = q_one
#     weight[label == 0] = q_zero
#     n = len(weight)
#     m = len(weight[0])
#     for i in range(n):
#         for j in range(1, m):
#             if weight[i][j] != weight[i][j - 1]:
#                 for count in range(1):
#                     if j + count < m and j - count - 1 >= 0:
#                         weight[i][j + count] *= ratio - (0.1 * count)
#                         weight[i][j - count - 1] *= ratio - (0.1 * count)
#                 break
#     return weight


def pre_emphasis(signal, alpha=0.97):
    # 使用roll函数将最后一维向右移动一个位置
    emphasized_signal = signal - alpha * torch.roll(signal, shifts=1, dims=-1)
    # 将最后一维的第一个元素恢复为原始值，因为它没有前一个值可以减去
    emphasized_signal[:, :, 0] = signal[:, :, 0]
    return emphasized_signal


def initialize_weights_zero(m):
    if isinstance(m, nn.Linear):
        init.constant_(m.weight, 0)
        if m.bias is not None:
            init.constant_(m.bias, 0)


def initialize_weights_uniform(m):
    if isinstance(m, nn.Linear):
        init.uniform_(m.weight, a=-0.1, b=0.1)
        if m.bias is not None:
            init.uniform_(m.bias, a=-0.1, b=0.1)


def initialize_weights_normal(m):
    if isinstance(m, nn.Linear):
        init.normal_(m.weight, mean=0, std=0.01)
        if m.bias is not None:
            init.normal_(m.bias, mean=0, std=0.01)


def initialize_weights_xavier(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)


def initialize_weights_he(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)


def reset_overlap(pred, overlap):
    for idx in range(1, pred.size(0)):
        tail = pred[idx - 1, -overlap:]
        head = pred[idx, :overlap]
        new_over_lap = torch.max(tail, head)
        pred[idx - 1, -overlap:] = new_over_lap
        pred[idx, :overlap] = new_over_lap


def high_pass_filter(data, sampling_freq, cutoff_freq):
    b, a = signal.butter(4, cutoff_freq / (sampling_freq / 2), 'high')
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data


def low_pass_filter(data, sampling_freq, cutoff_freq):

    b, a = signal.butter(4, cutoff_freq / (sampling_freq / 2), 'low')
    data = signal.filtfilt(b, a, data)

    return data


def process_data(label=True):
    for root, dirs, files in os.walk("unlabel"):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if "imu.csv" in file_path:
                    if label:
                        imu = pd.read_csv(file_path)
                    else:
                        imu = pd.read_csv(file_path, names="time::X::Y::Z".split("::"))
                    imu = imu.groupby('time').mean().reset_index()
                    imu['X'] = high_pass_filter(imu['X'], 1000, 40).round(3)
                    imu['Y'] = high_pass_filter(imu['Y'], 1000, 40).round(3)
                    imu['Z'] = high_pass_filter(imu['Z'], 1000, 40).round(3)
                    imu['time'] = imu['time'] - imu['time'].iloc[0]
                    imu.to_csv(file_path, index=False)
                elif "gas.csv" in file_path:
                    if label:
                        gas = pd.read_csv(file_path)
                    else:
                        gas = pd.read_csv(file_path, names="time::value".split("::"))
                    gas = gas.groupby('time').mean().reset_index()
                    gas['value'] = low_pass_filter(gas['value'], 100, 10).round(3)
                    gas['time'] = gas['time'] - gas['time'].iloc[0]
                    gas.to_csv(file_path, index=False)
            except Exception as e:
                print(f"Error in SegThread: {e}")
    print("finished")


def unlabel_data_process(folder_path):
    group_set = set()
    data = {"Dir": []}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_name, file_extension = os.path.splitext(file)
            file_prefix = file_name.split('_')[0]
            if file_prefix not in group_set:
                imu_file = f"{file_prefix}_imu.csv"
                gas_file = f"{file_prefix}_gas.csv"
                audio_file = f"{file_prefix}_vedio.wav"
                imu = pd.read_csv(os.path.join(root, imu_file))
                gas = pd.read_csv(os.path.join(root, gas_file))
                if imu.isnull().values.any():
                    print(f"Empty values found in IMU file: {root} {imu_file}")

                if gas.isnull().values.any():
                    print(f"Empty values found in Gas file: {root} {gas_file}")
                if os.path.exists(os.path.join(root, imu_file)) and os.path.exists(os.path.join(root, gas_file)) and\
                   os.path.exists(os.path.join(root, audio_file)):
                    group_set.add(file_prefix)
                    data["Dir"].append(os.path.join(root, audio_file))

    df = pd.DataFrame(data)

    # 将DataFrame写入CSV文件
    df.to_csv("unlabel_meta.csv", header=None, index=False)


def process_folder(root_folder):
    files_list = []

    for root, dirs, files in os.walk(root_folder):
        for file_name in files:
            if 'Annotated.json' in file_name:
                files_list.append([root])

    with open('healthy_meta.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for row in files_list:
            csv_writer.writerow(row)


def count_patient(root_folder):
    patient_set = set()
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if "Annotated" in filename:
                patient_name = foldername.split('/')[-1].split('_')[0]
                patient_set.add(patient_name)
    for name in patient_set:
        print(name)
    print(len(patient_set))


def count_swallow(root_folder):
    swallow_count = 0
    total_duration = 0
    swallow_duration = []
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if "Annotated" in filename:
                Annotated_path = os.path.join(foldername, "Annotated.json")
                gas_path = os.path.join(foldername, "gas.csv")
                with open(Annotated_path, 'r') as json_file:
                    data = json.load(json_file)
                label_list = data['label_list']
                for swallow_dict in label_list:
                    duration = (swallow_dict['end'] - swallow_dict['start']) * 1000
                    swallow_duration.append(duration)
                gas_times = pd.read_csv(gas_path)['time'].values
                total_duration += gas_times[-1] - gas_times[0]
                swallow_count += len(label_list)
                break
    print(f"total time={total_duration}, swallow ratio={sum(swallow_duration)/total_duration * 100}%")
    print(f"avg_duration={sum(swallow_duration)/len(swallow_duration)}")
    print(f"swallow count={swallow_count}")


def delete_files(root_folder, file_pattern):
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if file_pattern in filename:
                file_path = os.path.join(foldername, filename)
                os.remove(file_path)
                print(f"Deleted: {file_path}")


def split_data(meta_path):
    data = pd.read_csv(meta_path)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=4)
    test_data.to_csv('test_data.csv', index=False)
    train_data.to_csv('patient_train_data.csv', index=False)


def padding(signal, win_sample, hop_sample):
    total_length = signal.shape[1]
    num_windows = (total_length - win_sample) // hop_sample + 1
    new_total_length = num_windows * hop_sample + win_sample
    num_zeros_to_pad = new_total_length - total_length
    if num_zeros_to_pad > 0:
        signal = torch.nn.functional.pad(signal, (0, num_zeros_to_pad, 0, 0), mode='constant', value=0)
    return signal


# def bulid_config():
#     config_dict = {
#         'AudioSampleRate': 16000,
#         'ImuSampleRate': 1000,
#         'GasSampleRate': 100,
#         'BinNum': 15,
#         'WinDuration': 3,
#         'HopDuration': 2,
#         'BatchSize': 100,
#         'TestBatchSize': 300,
#         'Epochs': 500,
#         'LearningRate': 0.1,
#         'Seed': 1,
#         'Cuda': 1,
#         'NumWorkers': 2,
#         'LabelMeta': 'patient_meta.csv',
#         'Splits': 10,
#         'Model': 'AttnSleep',
#         "Momentum": 0.5,
#         "LogInterval": 100,
#         "DisableGpu": False
#         }
#     with open('config.json', 'w') as json_file:
#         json_file.write(json.dumps(config_dict, indent=4, ensure_ascii=False))

def frame(signal, win_sample, hop_sample):
    num_channels = signal.shape[0]
    signal_len = signal.shape[1]
    num_frames = (signal_len - win_sample) // hop_sample + 1
    frames = torch.zeros((num_frames, num_channels, win_sample))

    for i in range(num_frames):
        start = i * hop_sample
        end = start + win_sample
        frames[i, :, :] = signal[:, start:end]

    return frames


def frame_cut(frame_length, label_list, win_duration, hop_duartion, bins_num):
    labels = []
    bin_s = win_duration / bins_num
    labels = [[0] * bins_num for _ in range(frame_length)]
    for label in label_list:
        start_value = label['start']
        end_value = label['end']
        for i in range(frame_length):
            for bin in range(bins_num):
                bin_onset = i * hop_duartion + bin * bin_s
                bin_offset = bin_onset + bin_s
                if bin_onset > end_value or bin_offset < start_value:
                    pass
                else:
                    labels[i][bin] = 1.0
    return labels


def mask_cut(labels, win_duration, hop_duartion, bins_num):
    p0 = 0.75
    p1 = 0.25
    weight_0 = 1 / p0
    weight_1 = 1 / p1
    bin_duration = win_duration / bins_num
    hop_samples = hop_duartion / bin_duration
    over_lap = int(bins_num - hop_samples)
    label_1d = labels[0].copy()
    for idx in range(1, len(labels)):
        label_1d += labels[idx][over_lap:]

    # 找出所有连续的1开始的索引到结束的索引
    start_indexes = []
    end_indexes = []
    in_sequence = False
    for i, num in enumerate(label_1d):
        if num == 1 and not in_sequence:
            start_indexes.append(i)
            in_sequence = True
        elif num == 0 and in_sequence:
            end_indexes.append(i - 1)
            in_sequence = False
    # 如果连续的1一直到列表结束，加上最后一个索引
    if in_sequence:
        end_indexes.append(len(label_1d) - 1)

    # 打印结果
    # print("连续的1开始的索引到结束的索引:")
    label_1d = [weight_0 for _ in label_1d]
    for start, end in zip(start_indexes, end_indexes):
        label_1d = mask_normal(label_1d, weight_1, start, end)
    masks = []
    start = 0
    for i in range(len(labels)):
        masks.append(label_1d[start:start + bins_num])
        start += bins_num - over_lap
    return torch.Tensor(masks)


def generate_reverse_mask(data_size, std_factor=0.4):
    # 生成一维正态分布的蒙版并反转
    x = np.linspace(-1, 1, data_size)
    # mask = np.random.exponential(scale=std_factor)
    # mask = mask * 2 * std_factor + 0.5
    mask = np.exp(-x**2 / (2 * std_factor**2))  # 调整标准差
    mask = 1 - mask  # 反转正态分布
    return mask


def generate_normalize_mask(data_size):
    # 生成一维正态分布的蒙版并反转
    x = np.linspace(-1, 1, data_size)
    mask = np.exp(-x**2)
    return mask


def mask_normal(label, weight_1, start_idx, end_idx):
    data_size = end_idx - start_idx + 1  # 数据大小
    label_np = np.array(label)
    label_np[start_idx:end_idx + 1] = weight_1

    mask = generate_reverse_mask(data_size)
    # mask = generate_normalize_mask(data_size)

    data = label_np[start_idx:end_idx + 1]
    scaled_array = mask * (np.sum(data) / np.sum(mask))
    label_np[start_idx:end_idx + 1] = scaled_array
    return label_np.tolist()


# def dump_data(audio_list, imu_list, gas_list, labels_list, masks_list, dump_dir):
#     stacked_audio = torch.cat(audio_list, dim=0)
#     stacked_imu = torch.cat(imu_list, dim=0)
#     stacked_gas = torch.cat(gas_list, dim=0)
#     stacked_labels = torch.cat(labels_list, dim=0)
#     stacked_masks = torch.cat(masks_list, dim=0)
#     count = 0
#     length = 8
#     timestamp = int(time.time())  # 获取当前时间戳
#     for i in range(0, stacked_audio.shape[0], length):
#         subset_audio = stacked_audio[i: i + length]
#         subset_imu = stacked_imu[i: i + length]
#         subset_gas = stacked_gas[i: i + length]
#         subset_labels = stacked_labels[i: i + length]
#         subset_masks = stacked_masks[i: i + length]
#         sample_dict = {'audio': subset_audio, 'imu': subset_imu, 'gas': subset_gas, 'labels': subset_labels, 'masks': subset_masks}
#         # 保存字典到文件
#         output_file = f'Aspirat/data/{dump_dir}/sample_dict_{timestamp}_{count}.pt'
#         torch.save(sample_dict, output_file)
#         print(f"Shape:{stacked_audio.shape[0]} i={i} Dump {count} file={output_file}")
#         count += 1


def mix(signal):
    if signal.shape[0] == 2:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal


def resample(signal, sr, target_sample_rate):
    if sr != target_sample_rate:
        resample = torchaudio.transforms.Resample(sr, target_sample_rate)
        signal = resample(signal)
    return signal


def predict_calculate_overlap(model, root_path, data_list, pth_path, device):
    model.load_state_dict(torch.load(pth_path))
    model.eval()
    overlap_list = []
    for file in data_list:
        file_path = os.path.join(root_path, file)
        loaded_dict = torch.load(file_path)
        audio = loaded_dict['audio'].to(device)
        imu = loaded_dict['imu'].to(device)
        gas = loaded_dict['gas'].to(device)
        label = loaded_dict['labels'].to(device)
        output = model(audio, imu, gas)
        events = Advanced_post_processing(output)
        label_events = Advanced_post_processing(label)
        for event in label_events:
            overlap_rate = 0
            for pre_event in events:
                overlap_rate = max(overlap_rate, calculate_jaccard_index(pre_event[0], pre_event[1], event[0], event[1]))
            overlap_list.append(overlap_rate)
    return overlap_list


def Advanced_post_processing(pred, hop_duration=2, bin_duration=0.2):
    events = []
    swallow_flag = False
    for i in range(pred.shape[0]):
        pred_win = pred[i, :]
        for j in range(pred_win.shape[0]):
            bin_predicted = pred_win[j].item()
            if swallow_flag:
                if bin_predicted >= 0.4:
                    start_point = i * hop_duration + j * bin_duration
                    end_point = i * hop_duration + (j + 1) * bin_duration
                    events.append([start_point, end_point])
                else:
                    swallow_flag = False
            else:
                if bin_predicted >= 0.6:
                    swallow_flag = True
                    start_point = i * hop_duration + j * bin_duration
                    end_point = i * hop_duration + (j + 1) * bin_duration
                    events.append([start_point, end_point])
    events.sort(key=lambda x: x[0])
    max_silence = 0.4
    min_dur = 0.3

    merge_silence_events(events, max_silence)
    del_min_duration_events(events, min_dur)

    for i in range(len(events)):
        events[i][0] = round(events[i][0], 3)
        events[i][1] = round(events[i][1], 3)

    events.sort(key=lambda x: x[0])
    return events


def del_min_duration_events(events, min_duration):
    count = 0
    while count < len(events) - 1:
        if events[count][1] - events[count][0] < min_duration:
            del events[count]
        else:
            count += 1
    if len(events) > 0 and events[count][1] - events[count][0] < min_duration:
        del events[count]


def compute_overlap_ratio(p_start, p_end, t_start, t_end):
    overlap_start = max(p_start, t_start)
    overlap_end = min(p_end, t_end)

    if overlap_start >= overlap_end:
        return 0.0

    predicted_duration = p_end - p_start
    true_duration = t_end - t_start
    overlap_duration = overlap_end - overlap_start

    overlap_ratio = overlap_duration / (predicted_duration + true_duration - overlap_duration)

    return overlap_ratio


def calculate_jaccard_index(a, b, x, y):
    """
    计算两个时间段的 Jaccard 相似系数。

    参数:
    a, b: 第一个时间段的开始和结束时间
    x, y: 第二个时间段的开始和结束时间

    返回:
    Jaccard 相似系数
    """
    # 确保时间段是有效的
    if a > b or x > y:
        raise ValueError("Invalid time periods")

    # 计算交集的开始和结束时间
    intersection_start = max(a, x)
    intersection_end = min(b, y)

    # 如果没有交集
    if intersection_start >= intersection_end:
        return 0.0

    # 计算交集和并集的持续时间
    intersection_duration = intersection_end - intersection_start
    union_duration = (b - a) + (y - x) - intersection_duration

    # 计算 Jaccard 相似系数
    jaccard_index = intersection_duration / union_duration

    return jaccard_index


def merge_silence_events(events, min_silence):
    count = 0
    while count < len(events) - 1:
        if (events[count][1] >= events[count + 1][0]) or (events[count + 1][0] - events[count][1] <= min_silence):
            events[count][1] = max(events[count + 1][1], events[count][1])
            del events[count + 1]
        else:
            count += 1


def merge_rows_and_remove_overlap(input):
    two_d_list = input.tolist()
    if not two_d_list:
        return []

    merged_list = []

    for i in range(len(two_d_list) - 1):
        # Current row
        current_row = two_d_list[i]
        # Next row
        next_row = two_d_list[i + 1]

        # Handle the overlapping part
        for j in range(-5, 0):
            next_row[j + 5] = (current_row[j] + next_row[j + 5]) / 2

        # Append the non-overlapping part of the current row
        merged_list.extend(current_row[:-5])

    # Append the last row in full
    merged_list.extend(two_d_list[-1])

    return merged_list


def post_process(input_data, rebuild=True):
    sequence = merge_rows_and_remove_overlap(input_data)
    upper_limit = 0.6
    lower_limit = 0.4
    swallow_status = False
    output_sequence = []
    threshold = 0.5  # Threshold for checking the previous value
    for i, point in enumerate(sequence):
        if point > upper_limit:
            if i > 0 and output_sequence[-1] == 0 and sequence[i-1] > threshold:
                # Modify the previous point if it exceeds the threshold
                output_sequence[-1] = 1
            swallow_status = True
        elif point < lower_limit:
            swallow_status = False

        if swallow_status:
            output_sequence.append(1)
        else:
            output_sequence.append(0)
    delete_short_event(output_sequence)
    if rebuild:
        return rebuild_label(output_sequence)
    else:
        return output_sequence


def rebuild_label(sequnce):
    ouput = []
    for i in range(0, len(sequnce) - 14, 10):
        ouput.append(sequnce[i:i + 15])
    return torch.tensor(ouput)


def data_build(healty=False):
    with open('Aspirat/config.json', 'r') as json_file:
        config_dict = json.load(json_file)
    meta_csv = config_dict['patient_data_meta']
    dump_dir = config_dict['patient_data_dir']
    if healty:
        meta_csv = config_dict['healthy_data_meta']
        dump_dir = config_dict['healthy_data_dir']
    data = pd.read_csv(meta_csv)
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
    bins_num = config_dict['BinNum']
    # audio_list, imu_list, gas_list, labels_list, masks_list = [], [], [], [], []
    for root_path in data.iloc[:, 0]:
        audio_path = root_path + '/audio.wav'
        imu_path = root_path + '/imu.csv'
        gas_path = root_path + '/gas.csv'
        Annotated_path = root_path + '/Annotated.json'
        with open(Annotated_path, 'r') as json_file:
            data = json.load(json_file)
        label_list = data['label_list']
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
        labels = frame_cut(len(audio_frames), label_list, win_duration, hop_duartion, bins_num)
        labels = torch.Tensor(labels)
        if not (audio_frames.shape[0] == imu_frames.shape[0] == gas_frames.shape[0]):
            min_length = min(audio_frames.shape[0], imu_frames.shape[0], gas_frames.shape[0])
            audio_frames = audio_frames[:min_length, :, :]
            imu_frames = imu_frames[:min_length, :, :]
            gas_frames = gas_frames[:min_length, :, :]
            labels = labels[:min_length]
        assert not torch.isnan(audio_frames).any() and not torch.isnan(imu_frames).any() and not torch.isnan(gas_frames).any()
        sample_dict = {'audio': audio_frames, 'imu': imu_frames, 'gas': gas_frames, 'labels': labels}
        output_file = f'Aspirat/data/{dump_dir}/{root_path.split("/")[-1]}.pt'
        torch.save(sample_dict, output_file)
        print(f"Shape:{audio_frames.shape[0]} file={output_file}")
        # audio_list.append(audio_frames)
        # imu_list.append(imu_frames)
        # gas_list.append(gas_frames)
        # labels_list.append(labels)
        # masks_list.append(masks_tensor)
        # if len(audio_list) >= 20:
        #     dump_data(audio_list, imu_list, gas_list, labels_list, masks_list, dump_dir)
        #     audio_list, imu_list, gas_list, labels_list, masks_list = [], [], [], [], []
    # dump_data(audio_list, imu_list, gas_list, labels_list, masks_list, dump_dir)
    print("Data Transform Finished!")


def rebuild_target(data_list):
    window_size = 15
    stride = 10
    frames = []

    # 提取窗口并将其转换为PyTorch张量
    for i in range(0, len(data_list) - window_size + 1, stride):
        window = data_list[i:i + window_size]
        frames.append(window)

    # 转换成PyTorch张量
    return torch.tensor(frames)


def delete_short_event(data):
    continue_one = process_type(data=data, distinct_val=0, continue_val=1)
    continue_zero = process_type(data=continue_one, distinct_val=1, continue_val=0)
    return continue_zero


def process_type(data, distinct_val=0, continue_val=1, count=2):
    n = len(data)
    i = 0
    result = data[:]
    while i < n:
        if result[i] == distinct_val:
            start = i
            while i < n and result[i] == distinct_val:
                i += 1
            end = i

            if start > 0 and end < n and result[start - 1] == continue_val and result[end] == continue_val and end - start < count + 1:
                for j in range(start, end):
                    result[j] = continue_val
        else:
            i += 1
    return result


def custom_sort(file):
    split_name = file.split("_")
    time_stamp = split_name[-2]
    count = split_name[-1].split(".")[-2]
    return int(time_stamp+count)


def resort_meta():
    with open('Aspirat/config.json', 'r') as json_file:
        config_dict = json.load(json_file)
    data = os.listdir(f"Aspirat/{config_dict['train_dir']}")
    data.sort(key=custom_sort)
    df = pd.DataFrame(data)
    df.to_csv(f"Aspirat/{config_dict['train_dir']}/meta.csv", index=False, header=False)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None, name="", count=0):
    for i, spec in enumerate(specgram):
        if ax is None:
            _, ax = plt.subplots(1, 1)
        if title is not None:
            ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.imshow(librosa.power_to_db(spec.cpu()), origin="lower", aspect="auto", interpolation="nearest")
        filename = f"picture/{name}_{count}_{i}.png"
        # 保存图像到本地文件
        plt.savefig(filename)
        # 显示保存成功的信息
        print(f'Mel spectrogram saved as {filename}')
    plt.close()


# def extract_feature(healthy=False):
#     with open('Aspirat/config.json', 'r') as json_file:
#         config_dict = json.load(json_file)
#     meta_csv = config_dict['patient_data_meta']
#     if healthy:
#         meta_csv = config_dict['healthy_data_meta']
#     data = pd.read_csv(meta_csv)
#     audio_sample_rate = config_dict['AudioSampleRate']
#     imu_sample_rate = config_dict['ImuSampleRate']
#     gas_sample_rate = config_dict['GasSampleRate']
#     features_list = []
#     for root_path in data.iloc[:, 0]:
#         audio_path = root_path + '/audio.wav'
#         imu_path = root_path + '/imu.csv'
#         gas_path = root_path + '/gas.csv'
#         Annotated_path = root_path + '/Annotated.json'
#         with open(Annotated_path, 'r') as json_file:
#             data = json.load(json_file)
#         label_list = data['label_list']
#         audio, sr = librosa.load(audio_path, sr=audio_sample_rate)
#         if audio.ndim > 1:
#             audio = np.mean(audio, axis=0)
#         imu = pd.read_csv(imu_path)
#         x_values = imu['X'].values
#         y_values = imu['Y'].values
#         z_values = imu['Z'].values
#         # 将 x、y、z 列的数值组合成一个三通道的数组
#         imu = np.column_stack((x_values, y_values, z_values))
#         gas = np.array(pd.read_csv(gas_path)['value'].values)
#         for count, label in enumerate(label_list):
#             start_value = label['start']
#             end_value = label['end']
#             audio_start = int(start_value * audio_sample_rate)
#             audio_end = int(end_value * audio_sample_rate)
#             imu_start = int(start_value * imu_sample_rate)
#             imu_end = int(end_value * imu_sample_rate)
#             gas_start = int(start_value * gas_sample_rate)
#             gas_end = int(end_value * gas_sample_rate)
#             sample_audio = audio[audio_start:audio_end]
#             sample_imu = imu[imu_start:imu_end, :]
#             sample_gas = gas[gas_start:gas_end]
#             features = extract_time_domain_features("audio", sample_audio)
#             features.update(extract_frequency_domain_features("audio", sample_audio, audio_sample_rate))
#             features.update(extract_time_domain_features("imu_x", sample_imu[:, 0]))
#             features.update(extract_frequency_domain_features("imu_x", sample_imu[:, 0], imu_sample_rate))
#             features.update(extract_time_domain_features("imu_y", sample_imu[:, 1]))
#             features.update(extract_frequency_domain_features("imu_y", sample_imu[:, 1], imu_sample_rate))
#             features.update(extract_time_domain_features("imu_z", sample_imu[:, 2]))
#             features.update(extract_frequency_domain_features("imu_z", sample_imu[:, 2], imu_sample_rate))
#             features.update(extract_time_domain_features("gas", sample_gas))
#             features.update(extract_frequency_domain_features("gas", sample_gas, gas_sample_rate))
#             features_list.append(features)
#         print(f"file {root_path} extract success.")
#     df = pd.DataFrame(features_list)
#     csv_file_path = 'patient_features.csv'
#     df.to_csv(csv_file_path, index=False)
#     print(f"保存的特征 DataFrame: healthy={healthy}")


# def extract_frequency_domain_features(feature_type, data, sampling_rate):
#     # 计算信号的功率谱密度
#     if data.shape[0] < 256:
#         frequencies, power_density = welch(data, fs=sampling_rate, nperseg=64)
#     else:
#         frequencies, power_density = welch(data, fs=sampling_rate)

#     # 找到功率谱密度最大值对应的频率（主频率）
#     dominant_frequency = frequencies[np.argmax(power_density)]

#     # 计算频谱能量
#     spectral_energy = np.sum(power_density)

#     # 计算频谱中心
#     spectral_centroid = np.sum(frequencies * power_density) / np.sum(power_density)

#     # 计算频谱带宽
#     spectral_bandwidth = np.sqrt(np.sum((frequencies - spectral_centroid)**2 * power_density) / np.sum(power_density))

#     # 计算频谱斜度
#     spectral_skewness = np.sum(((frequencies - spectral_centroid) / spectral_bandwidth)**3 * power_density) / np.sum(power_density)

#     # 计算频谱峰度
#     spectral_kurtosis = np.sum(((frequencies - spectral_centroid) / spectral_bandwidth)**4 * power_density) / np.sum(power_density)

#     # 将功率谱密度数组转换成字符串
#     power_density_str = ','.join(map(str, power_density))

#     # 返回提取的频域特征
#     features = {
#         f'{feature_type}_dominant_frequency': dominant_frequency,
#         f'{feature_type}_spectral_energy': spectral_energy,
#         f'{feature_type}_spectral_centroid': spectral_centroid,
#         f'{feature_type}_spectral_bandwidth': spectral_bandwidth,
#         f'{feature_type}_spectral_skewness': spectral_skewness,
#         f'{feature_type}_spectral_kurtosis': spectral_kurtosis,
#         f'{feature_type}_power_density': power_density_str  # 可选：返回完整的功率谱密度数组
#     }

#     return features


# def extract_time_domain_features(feature_type, data):
#     # 计算均值、标准差、最大值、最小值
#     mean_value = np.mean(data)
#     std_deviation = np.std(data)
#     max_value = np.max(data)
#     min_value = np.min(data)

#     # 计算峰值（绝对值的最大值）
#     peak_value = np.max(np.abs(data))

#     # 计算偏度和峰度
#     data_skewness = skew(data)
#     data_kurtosis = kurtosis(data)

#     # 返回提取的时域特征
#     features = {
#         f'{feature_type}_mean': mean_value,
#         f'{feature_type}_std_deviation': std_deviation,
#         f'{feature_type}_max': max_value,
#         f'{feature_type}_min': min_value,
#         f'{feature_type}_peak': peak_value,
#         f'{feature_type}_skewness': data_skewness,
#         f'{feature_type}_kurtosis': data_kurtosis
#     }

#     return features


if __name__ == "__main__":
    count_patient("../SegmentSwallow/patient")
    # count_swallow("../SegmentSwallow/patient")
    # extract_feature()
    # data_build(healty=True)
    # resort_meta()
    # process_folder("SegmentSwallow/healthy")
    # split_data('patient_meta.csv')
    # delete_files('SegmentSwallow', '.mp4')
    # delete_files('SegmentSwallow', 'denoise')
    # data_tensor = torch.tensor([[0, 0.2, 0.3, 0, 0, 0, 0.4, 0.5, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
    #                             [0.5, 0.46, 0.3, 0, 0, 0, 0.4, 0.5, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
    #                             [0.5, 0.2, 0.3, 0, 0, 0.6, 0.46, 0.5, 0.5, 0.6, 0.6, 0.5, 0.4, 0.3, 0.45]])

    # post_process(data_tensor)
    pass
