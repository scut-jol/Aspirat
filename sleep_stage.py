import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Squeeze-and-Excitation block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv1d(channel, channel // reduction, kernel_size=1)
        self.fc2 = nn.Conv1d(channel // reduction, channel, kernel_size=1)

    def forward(self, x):
        se = x.mean(-1, keepdim=True)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se

# Define the MSCNN block
class MSCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pool_size):
        super(MSCNNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.maxpool = nn.MaxPool1d(kernel_size=pool_size)
        self.dropout = nn.Dropout(0.5)
        self.se_block = SEBlock(out_channels)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.se_block(x)
        return x

# Define the full model
class SleepStageModel(nn.Module):
    def __init__(self):
        super(SleepStageModel, self).__init__()
        self.small_scale_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=3),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Dropout(0.1),
            nn.Conv1d(32, 32, kernel_size=7, stride=2),
            nn.Conv1d(32, 32, kernel_size=7, stride=2),
            nn.Conv1d(32, 32, kernel_size=7, stride=2),
            nn.MaxPool1d(kernel_size=4, stride=2),
            SEBlock(32)
        )
        
        self.large_scale_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=18, stride=6),
            nn.MaxPool1d(kernel_size=12, stride=6),
            nn.Dropout(0.1),
            nn.Conv1d(32, 32, kernel_size=9, stride=3),
            nn.Conv1d(32, 32, kernel_size=9, stride=3),
            nn.Conv1d(32, 32, kernel_size=9, stride=3),
            nn.MaxPool1d(kernel_size=8, stride=4),
            SEBlock(32)
        )
        self.point = nn.Conv1d(32, 5, kernel_size=1, stride=1)
        self.fc = nn.Sequential(
            nn.Linear(3030, 800),
            nn.Linear(800, 400),
            nn.Linear(400, 15)
        )
        
        self.bi_gru = nn.GRU(input_size=32, hidden_size=16, num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, audio, imu, gas):
        imu = imu.reshape(imu.size(0), 1, imu.size(1) * imu.size(2))
        x = torch.cat((audio, imu, gas), dim=2)
        x_small = self.small_scale_branch(x)
        x_large = self.large_scale_branch(x)
        
        x = torch.cat((x_small, x_large), dim=2)
        x = x.permute(0, 2, 1)
        x, _ = self.bi_gru(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.point(x)
        x = x.reshape(x.size(0), -1)
        x = self.sigmoid(self.fc(x))
        return x
