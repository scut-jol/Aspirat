
import torch
from torch.utils.data import Dataset
import os


def collate_fn(data):
    return data[0]


class Aspiration(Dataset):
    q_zero = 1
    q_one = 1

    def __init__(self, data_list, root_path, batch_size=32) -> None:
        super().__init__()
        self.audio_list = []
        self.imu_list = []
        self.gas_list = []
        self.label_list = []
        self.batch_size = batch_size
        self.__load__(root_path, data_list)

    def get_nums(self):
        num_ones = torch.sum(self.label_list == 1).item()
        num_zeros = torch.sum(self.label_list == 0).item()
        return num_ones, num_zeros

    def __load__(self, root_path, data_list):
        self.data = []
        for index in range(len(data_list.iloc[:, 0])):
            file_path = os.path.join(root_path, data_list.iloc[index, 0])
            loaded_dict = torch.load(file_path)
            self.audio_list.append(loaded_dict['audio'])
            self.imu_list.append(loaded_dict['imu'])
            self.gas_list.append(loaded_dict['gas'])
            self.label_list.append(loaded_dict['labels'])
        self.audio_list = torch.cat(self.audio_list, dim=0)
        self.imu_list = torch.cat(self.imu_list, dim=0)
        self.gas_list = torch.cat(self.gas_list, dim=0)
        self.label_list = torch.cat(self.label_list, dim=0)

    def __len__(self):
        return self.audio_list.shape[0] // self.batch_size

    def __getitem__(self, index):
        pos = index * self.batch_size
        audio = self.audio_list[pos: pos + self.batch_size, :, :]
        imu = self.imu_list[pos: pos + self.batch_size, :, :]
        gas = self.gas_list[pos: pos + self.batch_size, :, :]
        label = self.label_list[pos: pos + self.batch_size, :]
        return audio, imu, gas, label
