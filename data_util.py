import os
import pandas as pd
from scipy import signal
import csv
from sklearn.model_selection import train_test_split
import json
import torchaudio
import torch
import numpy as np
import time
import librosa
import matplotlib.pyplot as plt

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


def compare_files(file1_path, file2_path, output_path):
    # 读取文件1的内容
    with open(file1_path, 'r') as file1:
        lines_file1 = set(file1.readlines())

    # 读取文件2的内容
    with open(file2_path, 'r') as file2:
        lines_file2 = set(file2.readlines())

    # 找到两个文件中不同的行
    different_lines = lines_file1.symmetric_difference(lines_file2)

    # 将不同的行写入输出文件
    with open(output_path, 'w') as output_file:
        output_file.writelines(sorted(different_lines))


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


def dump_data(audio_list, imu_list, gas_list, labels_list, masks_list, dump_dir):
    stacked_audio = torch.cat(audio_list, dim=0)
    stacked_imu = torch.cat(imu_list, dim=0)
    stacked_gas = torch.cat(gas_list, dim=0)
    stacked_labels = torch.cat(labels_list, dim=0)
    stacked_masks = torch.cat(masks_list, dim=0)
    count = 0
    length = 8
    timestamp = int(time.time())  # 获取当前时间戳
    for i in range(0, stacked_audio.shape[0], length):
        subset_audio = stacked_audio[i: i + length]
        subset_imu = stacked_imu[i: i + length]
        subset_gas = stacked_gas[i: i + length]
        subset_labels = stacked_labels[i: i + length]
        subset_masks = stacked_masks[i: i + length]
        sample_dict = {'audio': subset_audio, 'imu': subset_imu, 'gas': subset_gas, 'labels': subset_labels, 'masks': subset_masks}
        # 保存字典到文件
        output_file = f'Aspirat/data/{dump_dir}/sample_dict_{timestamp}_{count}.pt'
        torch.save(sample_dict, output_file)
        print(f"Shape:{stacked_audio.shape[0]} i={i} Dump {count} file={output_file}")
        count += 1

def mix(signal):
    if signal.shape[0] == 2:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal


def resample(signal, sr, target_sample_rate):
    if sr != target_sample_rate:
        resample = torchaudio.transforms.Resample(sr, target_sample_rate)
        signal = resample(signal)
    return signal


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
    audio_list, imu_list, gas_list, labels_list, masks_list = [], [], [], [], []
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
        masks_tensor = mask_cut(labels, win_duration, hop_duartion, bins_num)
        labels = torch.Tensor(labels)
        if not (audio_frames.shape[0] == imu_frames.shape[0] == gas_frames.shape[0]):
            min_length = min(audio_frames.shape[0], imu_frames.shape[0], gas_frames.shape[0])
            audio_frames = audio_frames[:min_length, :, :]
            imu_frames = imu_frames[:min_length, :, :]
            gas_frames = gas_frames[:min_length, :, :]
            labels = labels[:min_length]
            masks_tensor = masks_tensor[:min_length]
        assert not torch.isnan(audio_frames).any() and not torch.isnan(imu_frames).any() and not torch.isnan(gas_frames).any()
        audio_list.append(audio_frames)
        imu_list.append(imu_frames)
        gas_list.append(gas_frames)
        labels_list.append(labels)
        masks_list.append(masks_tensor)
        if len(audio_list) >= 20:
            dump_data(audio_list, imu_list, gas_list, labels_list, masks_list, dump_dir)
            audio_list, imu_list, gas_list, labels_list, masks_list = [], [], [], [], []
        print(f"{root_path} file transformed!")
    dump_data(audio_list, imu_list, gas_list, labels_list, masks_list, dump_dir)
    resort_meta()
    print("Data Transform Finished!")

def post_process(input_data):
    data = input_data.tolist()
    upper_limit = 0.55
    lower_limit = 0.45
    swallow = False
    start_idx = 10
    over_lap =  5
    data_list = data[0][:10]
    for line in range(1, len(data)):
        for i in range(over_lap):
            data_list.append((data[line][i] + data[line - 1][start_idx + i]) / 2)
        for i in range(over_lap, start_idx):
            data_list.append(data[line][i])
    data_list.extend([value for value in data[-1][-5:]])
    for i in range(len(data_list)):
        if swallow:
            if data_list[i] >= lower_limit:
                data_list[i] = 1
            else:
                data_list[i] = 0
                swallow = False
        else:
            if data_list[i] >= upper_limit:
                data_list[i] = 1
                swallow = True
            else:
                data_list[i] = 0
    delete_short_event(data_list)
    return rebuild_target(data_list)

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
    counts = 4
    pre_type = 0
    for i in range(len(data)):
        if data[i] == pre_type:
            counts += 1
        else:
            pre_type = data[i]
            if counts <= 3:
                for idx in range(counts):
                    data[i - idx - 1] = pre_type
            counts = 1

def custom_sort(file):
    split_name = file.split("_")
    time_stamp  = split_name[-2]
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


if __name__ == "__main__":
    data_build(healty=False)
    resort_meta()
    # process_folder("SegmentSwallow/healthy")
    # split_data('patient_meta.csv')
    # delete_files('SegmentSwallow', '.mp4')
    # delete_files('SegmentSwallow', 'denoise')
    # data_tensor = torch.tensor([[0, 0.2, 0.3, 0, 0, 0, 0.4, 0.5, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
    #                             [0.5, 0.46, 0.3, 0, 0, 0, 0.4, 0.5, 0.4, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
    #                             [0.5, 0.2, 0.3, 0, 0, 0.6, 0.46, 0.5, 0.5, 0.6, 0.6, 0.5, 0.4, 0.3, 0.45]])

    # post_process(data_tensor)
    pass
