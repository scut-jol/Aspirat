import torch.nn as nn
import torch
from resnet import resnet34


class Encoder(nn.Module):
    def __init__(self, in_channel=1):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=16, kernel_size=3, padding='same', stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = nn.ModuleList()
        input_channel = 16
        for i in range(3):
            conv = nn.Conv2d(in_channels=input_channel, out_channels=input_channel * 2, kernel_size=3, stride=1, bias=False)
            bn = nn.BatchNorm2d(input_channel * 2)
            relu = nn.ReLU()
            input_channel *= 2
            self.layers.extend([conv, bn, relu])

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same', stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        for layer in self.layers:
            x = layer(x)

        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, batch_first=True, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        b, t, h = recurrent.size()
        t_rec = recurrent.reshape(t * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.reshape(b, t, -1)

        return output


class AspirationModel(nn.Module):
    def __init__(self):
        super(AspirationModel, self).__init__()

        self.audio_encoder = Encoder(1)  # resnet34(in_channel=1)
        self.imu_encoder = Encoder(3)  # resnet34(in_channel=3)
        self.gas_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=16, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(17, 3))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(1)
        self.max_pool_1d = nn.AdaptiveMaxPool1d(1)
        self.point_conv = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=3, padding='same', stride=1, bias=False)
        self.point_conv_1d = nn.Conv1d(in_channels=256, out_channels=2, kernel_size=3, stride=1, bias=False)
        reduction_ratio = 3
        atten_channel = 256
        self.atten_fc = nn.Sequential(
            nn.Linear(atten_channel, atten_channel // reduction_ratio),
            nn.ReLU(),
            nn.Linear(atten_channel // reduction_ratio, atten_channel),
            nn.BatchNorm1d(atten_channel),
            nn.ReLU(),
        )
        self.outer_bn = nn.BatchNorm1d(338)
        self.relu = nn.ReLU()
        self.rnn = nn.LSTM(260, 32, num_layers=2, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(21632, 32)
        self.sigmoid = nn.Sigmoid()
        # self.rnn = nn.Sequential(
        #     BidirectionalLSTM(10240, 1024, 128),
        #     BidirectionalLSTM(128, 32, 1))

    def attention(self, audio, imu, gas):
        audio_avg_out = self.atten_fc(self.avg_pool(audio).view(audio.size(0), -1))
        audio_max_out = self.atten_fc(self.max_pool(audio).view(audio.size(0), -1))
        imu_avg_out = self.atten_fc(self.avg_pool(imu).view(imu.size(0), -1))
        imu_max_out = self.atten_fc(self.max_pool(imu).view(imu.size(0), -1))
        gas_avg_out = self.atten_fc(self.avg_pool_1d(gas).view(imu.size(0), -1))
        gas_max_out = self.atten_fc(self.max_pool_1d(gas).view(imu.size(0), -1))

        audio = self.sigmoid(audio_avg_out + audio_max_out).unsqueeze(2).unsqueeze(3) * audio
        imu = self.sigmoid(imu_avg_out + imu_max_out).unsqueeze(2).unsqueeze(3) * imu
        gas = self.sigmoid(gas_avg_out + gas_max_out).unsqueeze(2) * gas

        return audio, imu, gas

    def outer_product(self, audio, imu, gas):
        audio = self.point_conv(audio)
        audio = audio.view(audio.size(0), -1)
        imu = self.point_conv(imu)
        imu = imu.view(imu.size(0), -1)
        gas = self.point_conv_1d(gas)
        gas = gas.view(gas.size(0), -1)
        abc_outer = torch.zeros((audio.shape[0], audio.shape[1], imu.shape[1], gas.shape[1]))
        abc_outer = torch.einsum('bi,bj,bk->bijk', audio, imu, gas)
        return abc_outer.view(abc_outer.size(0), 338, -1)

    def forward(self, audio, imu, gas):
        audio = self.audio_encoder(audio)
        imu = self.imu_encoder(imu)
        gas = self.gas_encoder(gas)

        audio, imu, gas = self.attention(audio, imu, gas)
        x = self.outer_product(audio, imu, gas)
        x = self.outer_bn(x)
        x = self.relu(x)
        assert not torch.isnan(x).any(), "x不在指定范围内"
        # x = x.permute(1, 0, 2)
        x, _ = self.rnn(x)
        x = self.outer_bn(x)
        x = self.relu(x)
        assert not torch.isnan(x).any(), "x不在指定范围内"
        # x = x.permute(1, 0, -1).squeeze()
        x = x.view(x.size(0), -1)
        x = self.linear(x).squeeze()
        x = self.sigmoid(x)
        return x


class AspirationResNetModel(nn.Module):
    def __init__(self, audio_channel=1, imu_channel=3, gas_channel=1):
        super(AspirationResNetModel, self).__init__()
        self.audio_encoder = resnet34(in_channel=audio_channel)
        self.imu_encoder = resnet34(in_channel=imu_channel)
        self.gas_encoder = nn.Sequential(
            nn.Conv1d(in_channels=gas_channel, out_channels=16, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(1)
        self.max_pool_1d = nn.AdaptiveMaxPool1d(1)
        self.zip_conv1 = nn.Conv2d(in_channels=384, out_channels=15, kernel_size=1)
        self.point_conv = nn.Conv2d(in_channels=512, out_channels=16, kernel_size=1)
        self.point_conv_1d = nn.Conv1d(in_channels=512, out_channels=16, kernel_size=1)
        reduction_ratio = 2
        atten_channel = 512
        self.atten_fc = nn.Sequential(
            nn.Linear(atten_channel, atten_channel // reduction_ratio),
            nn.ReLU(),
            nn.Linear(atten_channel // reduction_ratio, atten_channel),
            nn.LayerNorm(atten_channel),
            nn.ReLU()
        )
        self.outer_bn = nn.BatchNorm1d(15)
        self.relu = nn.ReLU()
        # self.rnn = nn.LSTM(512, 32, num_layers=2, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2048, 32)
        self.sigmoid = nn.Sigmoid()
        self.rnn = nn.Sequential(
            BidirectionalLSTM(8192, 1024, 512),
            BidirectionalLSTM(512, 128, 1))

    def attention(self, audio, imu, gas):
        avg_fea = self.avg_pool(audio)
        max_fea = self.max_pool(audio)
        avg_fea = avg_fea.view(audio.size(0), -1)
        max_fea = max_fea.view(audio.size(0), -1)
        audio_avg_out = self.atten_fc(avg_fea)
        audio_max_out = self.atten_fc(max_fea)
        avg_fea = self.avg_pool(imu)
        max_fea = self.max_pool(imu)
        avg_fea = avg_fea.view(imu.size(0), -1)
        max_fea = max_fea.view(imu.size(0), -1)
        imu_avg_out = self.atten_fc(avg_fea)
        imu_max_out = self.atten_fc(max_fea)
        avg_fea = self.avg_pool_1d(gas)
        max_fea = self.max_pool_1d(gas)
        avg_fea = avg_fea.view(gas.size(0), -1)
        max_fea = max_fea.view(gas.size(0), -1)
        gas_avg_out = self.atten_fc(avg_fea)
        gas_max_out = self.atten_fc(max_fea)

        audio = self.sigmoid(audio_avg_out + audio_max_out).unsqueeze(2).unsqueeze(3) * audio
        imu = self.sigmoid(imu_avg_out + imu_max_out).unsqueeze(2).unsqueeze(3) * imu
        gas = self.sigmoid(gas_avg_out + gas_max_out).unsqueeze(2) * gas

        return audio, imu, gas

    def outer_product(self, audio, imu, gas):
        audio = self.point_conv(audio).view(audio.size(0), -1)
        imu = self.point_conv(imu).view(imu.size(0), -1)
        gas = self.point_conv_1d(gas).view(gas.size(0), -1)

        abc_outer = torch.zeros((audio.shape[0], audio.shape[1], imu.shape[1], gas.shape[1]))
        abc_outer = torch.einsum('bi,bj,bk->bijk', audio, imu, gas)
        abc_outer = self.zip_conv1(abc_outer)
        return abc_outer.view(abc_outer.size(0), 15, -1)

    def forward(self, audio, imu, gas):
        audio = self.audio_encoder(audio)
        imu = self.imu_encoder(imu)
        gas = self.gas_encoder(gas)

        audio, imu, gas = self.attention(audio, imu, gas)
        x = self.outer_product(audio, imu, gas)
        x = self.outer_bn(x)
        x = self.relu(x)
        assert not torch.isnan(x).any(), "x不在指定范围内"
        x = self.rnn(x)
        assert not torch.isnan(x).any(), "x不在指定范围内"
        x = x.view(x.size(0), -1)
        x = self.sigmoid(x)
        return x