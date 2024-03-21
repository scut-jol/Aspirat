
import torch
from torch.utils.data import Dataset
import os

# class Aspiration(Dataset):
#     def __init__(self,
#                  meta_data,
#                  AUDIO_SAMPLE_RATE,
#                  IMU_SAMPLE_RATE,
#                  GAS_SAMPLE_RATE,
#                  win_duration,
#                  hop_duartion,
#                  bins_num,
#                  ) -> None:
#         super().__init__()

#         self.metaData = meta_data
#         self.win_duration = win_duration
#         self.hop_duartion = hop_duartion
#         self.audio_sample_rate = AUDIO_SAMPLE_RATE
#         self.audio_win_sample = int(win_duration * AUDIO_SAMPLE_RATE)
#         self.audio_hop_sample = int(hop_duartion * AUDIO_SAMPLE_RATE)
#         self.imu_sample_rate = IMU_SAMPLE_RATE
#         self.imu_win_sample = int(win_duration * IMU_SAMPLE_RATE)
#         self.imu_hop_sample = int(hop_duartion * IMU_SAMPLE_RATE)
#         self.gas_sample_rate = GAS_SAMPLE_RATE
#         self.gas_win_sample = int(win_duration * GAS_SAMPLE_RATE)
#         self.gas_hop_sample = int(hop_duartion * GAS_SAMPLE_RATE)
#         self.bins_num = bins_num
#         # self.label = label

#     def __len__(self):
#         return len(self.metaData)

#     def _resample(self, signal, sr, target_sample_rate):
#         if sr != target_sample_rate:
#             resample = torchaudio.transforms.Resample(sr, target_sample_rate)
#             signal = resample(signal)
#         return signal

#     def _mix(self, signal):
#         if signal.shape[0] == 2:
#             signal = torch.mean(signal, dim=0, keepdim=True)
#         return signal

#     def _padding(self, signal, win_sample, hop_sample):
#         total_length = signal.shape[1]
#         num_windows = (total_length - win_sample) // hop_sample + 1
#         new_total_length = num_windows * hop_sample + win_sample
#         num_zeros_to_pad = new_total_length - total_length
#         if num_zeros_to_pad > 0:
#             signal = torch.nn.functional.pad(signal, (0, num_zeros_to_pad, 0, 0), mode='constant', value=0)
#         return signal

#     def _frame(self, signal, win_sample, hop_sample):
#         num_channels = signal.shape[0]
#         signal_len = signal.shape[1]
#         num_frames = (signal_len - win_sample) // hop_sample + 1
#         frames = torch.zeros((num_frames, num_channels, win_sample))

#         for i in range(num_frames):
#             start = i * hop_sample
#             end = start + win_sample
#             frames[i, :, :] = signal[:, start:end]

#         return frames

#     def _frame_cut(self, frame_length, label_list):
#         labels = []
#         bin_s = self.win_duration / self.bins_num
#         labels = [[0] * self.bins_num for _ in range(frame_length)]
#         for label in label_list:
#             start_value = label['start']
#             end_value = label['end']
#             for i in range(frame_length):
#                 for bin in range(self.bins_num):
#                     bin_onset = i * self.hop_duartion + bin * bin_s
#                     bin_offset = bin_onset + bin_s
#                     if bin_onset > end_value or bin_offset < start_value:
#                         pass
#                     else:
#                         labels[i][bin] = 1.0
#         return torch.Tensor(labels)

#     def _get_label(self, path):
#         with open(path, 'r') as json_file:
#             data = json.load(json_file)
#         label_list = data['label_list']
#         return label_list

#     def __getitem__(self, index):
#         file_path = f"{self.metaData.iloc[index, 0]}/"
#         audio_path = file_path + 'audio.wav'
#         imu_path = file_path + 'imu.csv'
#         gas_path = file_path + 'gas.csv'
#         Annotated_path = file_path + 'Annotated.json'
#         label_list = self._get_label(Annotated_path)
#         audio, sr = torchaudio.load(audio_path)
#         imu = pd.read_csv(imu_path)
#         gas = pd.read_csv(gas_path)
#         imu_tensor = torch.tensor(imu[['X', 'Y', 'Z']].values, dtype=torch.float32).T
#         gas_tensor = torch.tensor(gas[['value']].values, dtype=torch.float32).T
#         audio = self._mix(audio)
#         audio = self._resample(audio, sr, self.audio_sample_rate)
#         audio = self._padding(audio, self.audio_win_sample, self.audio_hop_sample)
#         imu_tensor = self._padding(imu_tensor, self.imu_win_sample, self.imu_hop_sample)
#         gas_tensor = self._padding(gas_tensor, self.gas_win_sample, self.gas_hop_sample)
#         # 分窗
#         audio_frames = self._frame(audio, self.audio_win_sample, self.audio_hop_sample)
#         imu_frames = self._frame(imu_tensor, self.imu_win_sample, self.imu_hop_sample)
#         gas_frames = self._frame(gas_tensor, self.gas_win_sample, self.gas_hop_sample)
#         labels = self._frame_cut(len(audio_frames), label_list)
#         # 保证数据长度一致
#         if not (audio_frames.shape[0] == imu_frames.shape[0] == gas_frames.shape[0]):
#             min_length = min(audio_frames.shape[0], imu_frames.shape[0], gas_frames.shape[0])
#             audio_frames = audio_frames[:min_length, :, :]
#             imu_frames = imu_frames[:min_length, :, :]
#             gas_frames = gas_frames[:min_length, :, :]
#             labels = labels[:min_length]
#         assert not torch.isnan(audio_frames).any() and not torch.isnan(imu_frames).any() and not torch.isnan(gas_frames).any()
#         return {'audio': audio_frames, 'imu': imu_frames, 'gas': gas_frames, 'labels': labels}

def collate_fn(data):
    audio = torch.cat([data_dict['audio'] for data_dict in data], dim=0)
    imu = torch.cat([data_dict['imu'] for data_dict in data], dim=0)
    gas = torch.cat([data_dict['gas'] for data_dict in data], dim=0)
    labels = torch.cat([data_dict['labels'] for data_dict in data], dim=0)
    masks = torch.cat([data_dict['masks'] for data_dict in data], dim=0)
    return audio, imu, gas, labels, masks

class Aspiration(Dataset):
    def __init__(self, data_list, root_path) -> None:
        super().__init__()
        self.datas = data_list
        self.root_path = root_path

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        file_path = os.path.join(self.root_path, self.datas.iloc[index, 0])
        loaded_dict = torch.load(file_path)
        assert loaded_dict['audio'].shape[1] == 1 and loaded_dict['audio'].shape[2] == 48000
        assert loaded_dict['imu'].shape[1] == 3 and loaded_dict['imu'].shape[2] == 3000
        assert loaded_dict['gas'].shape[1] == 1 and loaded_dict['gas'].shape[2] == 300
        assert loaded_dict['labels'].shape[1] == 15 and loaded_dict['masks'].shape[1] == 15
        return loaded_dict
