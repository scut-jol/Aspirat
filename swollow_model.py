import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from copy import deepcopy
from conformer.conformer.model import Conformer
import torchaudio
from data_util import plot_spectrogram


########################################################################################
class SELayer2D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # channel_attn_cpu = y.detach().cpu()
        # torch.save(channel_attn_cpu, 'channel_attn_cpu.pt')
        return x * y.expand_as(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer2D(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class CoarseFine(nn.Module):
    def __init__(self, reduced_channel, kernel_size1, stride1, kernel_size2, stride2, in_channels=1):
        super(CoarseFine, self).__init__()
        drate = 0.1
        self.GELU = nn.GELU()
        self.hight_feature1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=kernel_size1, stride=stride1, bias=False, padding=stride1 - 1, dilation=1),
            nn.BatchNorm2d(64),
            self.GELU,
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
            nn.Dropout(drate)
        )
        self.hight_feature2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(128),
            self.GELU,
            nn.Conv2d(128, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(128),
            self.GELU,
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        )
        self.low_feature1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=kernel_size2, stride=stride2, bias=False, padding=stride2 - 1, dilation=1),
            nn.BatchNorm2d(64),
            self.GELU,
            nn.MaxPool2d(kernel_size=7, stride=2, padding=1),
            nn.Dropout(drate)
        )
        self.low_feature2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, bias=False, padding=2),
            nn.BatchNorm2d(128),
            self.GELU,
            nn.Conv2d(128, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(128),
            self.GELU,
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        )
        self.dropout = nn.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock2D, reduced_channel, 1)
        # self.no_AFR = nn.Conv2d(32, 30, kernel_size=3, stride=1, bias=False, padding=1)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_merge = self.hight_feature1(x)
        x_low = self.low_feature1(x)
        msg = F.interpolate(x_low, size=(x_merge.size(2), x_merge.size(3)), mode='bilinear', align_corners=False)
        x_merge = x_merge + msg

        x_merge = self.hight_feature2(x_merge)
        x_low = self.low_feature2(x_low)
        msg = F.interpolate(x_low, size=(x_merge.size(2), x_merge.size(3)), mode='bilinear', align_corners=False)
        x_merge = x_merge + msg
        x_merge = self.dropout(x_merge)

        x_merge = self.AFR(x_merge)
        return x_merge


class MRMSFFC1D(nn.Module):
    def __init__(self, reduced_channel, kernel_size1, stride1, kernel_size2, stride2, in_channels=1):
        super(MRMSFFC1D, self).__init__()
        drate = 0.5
        self.GELU = nn.GELU()  # for older versions of PyTorch.  For new versions use nn.GELU() instead.
        self.hight_feature1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=kernel_size1, stride=stride1, bias=False, padding=stride1 - 1, dilation=1),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=2),
            nn.Dropout(drate)
        )
        self.hight_feature2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=3, bias=False, padding=2),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1)
        )
        self.low_feature1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=kernel_size2, stride=stride2, bias=False, padding=stride2 - 1, dilation=1),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=12, stride=3, padding=2),
            nn.Dropout(drate)
        )
        self.low_feature2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=9, stride=3, bias=False, padding=2),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.Conv1d(128, 128, kernel_size=7, stride=3, bias=False, padding=2),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.MaxPool1d(kernel_size=5, stride=3, padding=1)
        )
        self.dropout = nn.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, reduced_channel, 1)
        self.no_AFR = nn.Conv1d(128, 30, kernel_size=3, stride=1, bias=False, padding=1)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_merge = self.hight_feature1(x)
        x_low = self.low_feature1(x)
        msg = F.interpolate(x_low, size=(x_merge.size(2)), mode='linear', align_corners=False)
        x_merge = x_merge + msg
        x_merge = self.hight_feature2(x_merge)
        x_low = self.low_feature2(x_low)
        msg = F.interpolate(x_low, size=(x_merge.size(2)), mode='linear', align_corners=False)
        x_merge = x_merge + msg
        x_merge = self.dropout(x_merge)
        x_merge = self.AFR(x_merge)
        # x_merge = self.no_AFR(x_merge)
        return x_merge


class MRCNN(nn.Module):
    def __init__(self, reduced_channel, kernel_size1, stride1, kernel_size2, stride2, in_channels=1):
        super(MRCNN, self).__init__()
        drate = 0.5
        self.GELU = nn.GELU()  # for older versions of PyTorch.  For new versions use nn.GELU() instead.
        self.features1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=kernel_size1, stride=stride1, bias=False, padding=stride1 - 1),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=kernel_size2, stride=stride2, bias=False, padding=stride2 - 1),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, reduced_channel, 1)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)
        return x_concat

##########################################################################################


def attention(query, key, value, dropout=None):
    "Implementation of Scaled dot product attention"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, afr_reduced_cnn_size, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.convs = clones(CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1), 3)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        "Implements Multi-head attention"
        nbatches = query.size(0)

        query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.convs[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.convs[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn = attention(query, key, value, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linear(x)


class SublayerOutput(nn.Module):
    '''
    A residual connection followed by a layer norm.
    '''

    def __init__(self, size, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TCE(nn.Module):
    '''
    Transformer Encoder

    It is a stack of N layers.
    '''

    def __init__(self, layer, N):
        super(TCE, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class EncoderLayer(nn.Module):
    '''
    An encoder layer

    Made up of self-attention and a feed forward layer.
    Each of these sublayers have residual and layer norm, implemented by SublayerOutput.
    '''
    def __init__(self, size, self_attn, feed_forward, afr_reduced_cnn_size, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_output = clones(SublayerOutput(size, dropout), 2)
        self.size = size
        self.conv = CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1, dilation=1)

    def forward(self, x_in):
        "Transformer Encoder"
        query = self.conv(x_in)
        x = self.sublayer_output[0](query, lambda x: self.self_attn(query, x_in, x_in))  # Encoder self-attention
        return self.sublayer_output[1](x, self.feed_forward)


class PositionwiseFeedForward(nn.Module):
    "Positionwise feed-forward network."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        "Implements FFN equation."
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SwollowModel1D(nn.Module):
    def __init__(self):
        super(SwollowModel1D, self).__init__()

        N = 1  # number of TCE clones
        signal_model = 1360
        d_ff = 120   # dimension of feed forward
        h = 5  # number of attention heads
        dropout = 0.1
        segment_length = 15
        reduced_channel = 30

        self.mrmsffc = MRMSFFC1D(reduced_channel, 26, 7, 50, 5, in_channels=1)
        signal_attn = MultiHeadedAttention(h, signal_model, reduced_channel)
        signal_ff = PositionwiseFeedForward(signal_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(signal_model, deepcopy(signal_attn), deepcopy(signal_ff), reduced_channel, dropout), N)
        self.relu = nn.ReLU()

        self.decoder = nn.Sequential(
            nn.Linear(signal_model * reduced_channel, segment_length * reduced_channel),
            self.relu,
            nn.Dropout(dropout),
            nn.Linear(segment_length * reduced_channel, segment_length),
            nn.Sigmoid()
        )

    def forward(self, audio, imu, gas):
        signal = torch.cat((audio, gas), dim=2)
        for i in range(3):
            imu_slice = torch.unsqueeze(imu[:, i, :], dim=1)
            signal = torch.cat((signal, imu_slice), dim=2)
        signal_feat = self.mrmsffc(signal)
        signal_features = self.tce(signal_feat)
        signal_features = signal_features.contiguous().view(signal_features.size(0), -1)
        final_output = self.decoder(signal_features)
        return final_output


class SwollowModel2D(nn.Module):
    def __init__(self):
        super(SwollowModel2D, self).__init__()

        N = 1  # number of TCE clones
        signal_model = 870
        d_ff = 120   # dimension of feed forward
        h = 6  # number of attention heads
        dropout = 0.1
        segment_length = 15
        reduced_channel = 32

        self.mrmsffc = CoarseFine(reduced_channel, 5, 1, 7, 3, in_channels=5)
        signal_attn = MultiHeadedAttention(h, signal_model, reduced_channel)
        signal_ff = PositionwiseFeedForward(signal_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(signal_model, deepcopy(signal_attn), deepcopy(signal_ff), reduced_channel, dropout), N)
        self.relu = nn.ReLU()
        self.decoder = nn.Sequential(
            nn.Linear(signal_model * reduced_channel, segment_length * reduced_channel),
            self.relu,
            nn.Dropout(dropout),
            nn.Linear(segment_length * reduced_channel, segment_length),
            nn.Sigmoid()
        )

    def forward(self, audio, imu, gas):
        signal = torch.cat((audio, imu, gas), dim=1)
        signal_feat = self.mrmsffc(signal)
        signal_feat = signal_feat.view(signal_feat.size(0), signal_feat.size(1), -1)
        signal_features = self.tce(signal_feat)
        signal_features = signal_features.contiguous().view(signal_features.size(0), -1)
        final_output = self.decoder(signal_features)
        return final_output


class LinerDecoder(nn.Module):
    def __init__(self, input_dim, ouput_dim):
        super(LinerDecoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, 30 * 8)
        self.bn1 = nn.BatchNorm1d(30 * 8)
        self.drop_out = nn.Dropout(0.1)
        self.fc3 = nn.Linear(30 * 8, ouput_dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc3(self.drop_out(x))
        x = self.sigmod(x)
        return x


class CFSCNet1D(nn.Module):
    def __init__(self):
        super(CFSCNet1D, self).__init__()
        segment_length = 15
        model_dim = 679
        reduced_channel = 64

        self.mrmsffc = MRMSFFC1D(reduced_channel, 26, 7, 50, 5, in_channels=1)
        self.conformer = Conformer(num_classes=segment_length,
                                   input_dim=model_dim,
                                   encoder_dim=reduced_channel,
                                   num_encoder_layers=1)
        self.decoder = LinerDecoder(960, segment_length)

    def forward(self, audio, imu, gas):
        imu = imu.reshape(imu.size(0), 1, imu.size(1) * imu.size(2))
        signal = torch.cat((audio, imu, gas), dim=2)
        signal = self.mrmsffc(signal)
        # signal = signal.reshape(signal.size(0), signal.size(1), -1)
        signal = self.conformer(signal)
        signal = signal.reshape(signal.size(0), -1)
        signal = self.decoder(signal)
        return signal


class CFSCNet(nn.Module):
    def __init__(self):
        super(CFSCNet, self).__init__()
        segment_length = 15
        reduced_channel = 64
        in_channels = 5
        model_dim = 174

        self.mrmsffc = CoarseFine(reduced_channel, 5, 1, 7, 3, in_channels=in_channels)
        self.conformer = Conformer(num_classes=segment_length,
                                   input_dim=model_dim,
                                   encoder_dim=reduced_channel,
                                   num_encoder_layers=1)
        self.decoder = LinerDecoder(960, segment_length)

        self.audio_tranform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=16,
            melkwargs={
                "n_fft": 2048,
                "n_mels": 64,
                "hop_length": 800,
                "mel_scale": "htk",
            },
        )
        self.imu_transform = torchaudio.transforms.MFCC(
            sample_rate=1000,
            n_mfcc=16,
            melkwargs={
                "n_fft": 256,
                "n_mels": 64,
                "hop_length": 50,
                "mel_scale": "htk",
            },
        )
        self.gas_transform = torchaudio.transforms.MFCC(
            sample_rate=100,
            n_mfcc=16,
            melkwargs={
                "n_fft": 26,
                "n_mels": 64,
                "hop_length": 5,
                "mel_scale": "htk",
            },
        )

    def forward(self, audio, imu, gas):
        audio = self.audio_tranform(audio)
        imu = self.imu_transform(imu)
        gas = self.gas_transform(gas)
        cnn_signal = torch.cat((audio, imu, gas), dim=1)

        cnn_signal = self.mrmsffc(cnn_signal)
        rnn_signal = cnn_signal.reshape(cnn_signal.size(0), cnn_signal.size(1), -1)
        rnn_signal = self.conformer(rnn_signal)
        rnn_signal = rnn_signal.reshape(rnn_signal.size(0), -1)
        ouput = self.decoder(rnn_signal)
        return ouput
######################################################################

class CFSCNetAttn(nn.Module):
    def __init__(self):
        super(CFSCNetAttn, self).__init__()
        segment_length = 15
        reduced_channel = 64
        in_channels = 5
        model_dim = 64

        self.mrmsffc = CoarseFine(reduced_channel, 3, 1, 7, 3, in_channels=in_channels)
        self.conformer = Conformer(num_classes=segment_length,
                                   input_dim=model_dim,
                                   encoder_dim=reduced_channel,
                                   num_encoder_layers=1)
        self.decoder = LinerDecoder(3264, segment_length)

        self.audio_tranform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=16,
            melkwargs={
                "n_fft": 2048,
                "n_mels": 64,
                "hop_length": 800,
                "mel_scale": "htk",
            },
        )
        self.imu_transform = torchaudio.transforms.MFCC(
            sample_rate=1000,
            n_mfcc=16,
            melkwargs={
                "n_fft": 256,
                "n_mels": 64,
                "hop_length": 50,
                "mel_scale": "htk",
            },
        )
        self.gas_transform = torchaudio.transforms.MFCC(
            sample_rate=100,
            n_mfcc=16,
            melkwargs={
                "n_fft": 26,
                "n_mels": 64,
                "hop_length": 5,
                "mel_scale": "htk",
            },
        )

    def forward(self, cnn_signal, audio=None, imu=None, gas=None):
        # audio = self.audio_tranform(audio)
        # imu = self.imu_transform(imu)
        # gas = self.gas_transform(gas)
        # cnn_signal = torch.cat((audio, imu, gas), dim=1)
        # cnn_signal_cpu = cnn_signal.detach().cpu()
        # torch.save(cnn_signal_cpu, 'cnn_signal_cpu.pt')

        cnn_signal = self.mrmsffc(cnn_signal)
        rnn_signal = cnn_signal.reshape(cnn_signal.size(0), cnn_signal.size(1), -1)
        rnn_signal = rnn_signal.permute(0, 2, 1)
        rnn_signal = self.conformer(rnn_signal)
        rnn_signal = rnn_signal.permute(0, 2, 1)
        rnn_signal = rnn_signal.reshape(rnn_signal.size(0), -1)
        ouput = self.decoder(rnn_signal)
        return ouput

class BotteleBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, cbam=False):
        super(BotteleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block=Bottleneck, num_blocks=[3, 4, 6, 3], num_classes=15):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.sigmod = nn.Sigmoid()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, audio, imu, gas):
        x = torch.cat((audio, imu, gas), dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmod(x)

        return x


class HybridCNNRNN(nn.Module):
    '''
     Estimation of laryngeal closure duration during swallowing without invasive X-rays
    '''
    def __init__(self):
        super(HybridCNNRNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=9, stride=3)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=9, stride=3)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=2)

        # RNN part (Bidirectional GRU)
        self.gru1 = nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=5, kernel_size=1, stride=1)
        # Fully connected layers
        self.fc1 = nn.Linear(7945, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 15)

        # Activation functions
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, audio, imu, gas):
        imu = imu.reshape(imu.size(0), 1, imu.size(1) * imu.size(2))
        x = torch.cat((audio, imu, gas), dim=2)
        # CNN part
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        # Flattening for RNN input
        x = x.permute(0, 2, 1)  # Reshape for GRU (batch, seq_len, feature)

        # RNN part
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)

        # Fully connected layers
        x = x.permute(0, 2, 1)
        x = self.conv3(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(self.leaky_relu(self.fc1(x)))
        x = self.dropout(self.leaky_relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))

        return x


class SwallowingMonitor(nn.Module):
    '''
    Deep Learning-Based Swallowing Monitor for Realtime Detection of Swallow Duration
    '''
    def __init__(self):
        super(SwallowingMonitor, self).__init__()
        self.audio_transform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=64,
            melkwargs={
                "n_fft": 1024,
                "n_mels": 64,
                "hop_length": 800,
                "mel_scale": "htk",
            },
        )
        self.imu_transform = torchaudio.transforms.MFCC(
            sample_rate=1000,
            n_mfcc=64,
            melkwargs={
                "n_fft": 128,
                "n_mels": 64,
                "hop_length": 50,
                "mel_scale": "htk",
            },
        )
        self.gas_transform = torchaudio.transforms.MFCC(
            sample_rate=100,
            n_mfcc=64,
            melkwargs={
                "n_fft": 128,
                "n_mels": 64,
                "hop_length": 5,
                "mel_scale": "htk",
            },
        )
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=3)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 1), stride=1)

        # Define the fully connected layer
        self.fc1 = nn.Linear(1280, 15)  # Adjust the input size based on the output of the conv layers

    def forward(self, audio, imu, gas):
        audio = self.audio_transform(audio)
        imu = self.imu_transform(imu)
        gas = self.gas_transform(gas)
        x = torch.cat((audio, imu, gas), dim=1)
        # Apply the convolutional and pooling layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        # Flatten the tensor
        x = torch.flatten(x, 1)

        # Apply softmax
        x = F.sigmoid(self.fc1(x))

        return x


class Transformer(nn.Module):
    def __init__(self, sqen_len=61, model_dim=64, num_heads=8, num_layers=2, output_dim=15, dropout=0.1):
        super(Transformer, self).__init__()
        # 输入嵌入层
        self.point_cnn = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1)

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.fc_out = nn.Sequential(
                nn.Linear(model_dim * sqen_len, model_dim),
                nn.Linear(model_dim, output_dim),
                nn.Sigmoid()
            )

        # 位置编码
        self.positional_encoding = self._generate_positional_encoding(model_dim)
        self.audio_transform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=64,
            melkwargs={
                "n_fft": 1024,
                "n_mels": 64,
                "hop_length": 800,
                "mel_scale": "htk",
            },
        )
        self.imu_transform = torchaudio.transforms.MFCC(
            sample_rate=1000,
            n_mfcc=64,
            melkwargs={
                "n_fft": 128,
                "n_mels": 64,
                "hop_length": 50,
                "mel_scale": "htk",
            },
        )
        self.gas_transform = torchaudio.transforms.MFCC(
            sample_rate=100,
            n_mfcc=64,
            melkwargs={
                "n_fft": 128,
                "n_mels": 64,
                "hop_length": 5,
                "mel_scale": "htk",
            },
        )

    def forward(self, audio, imu, gas):
        audio = self.audio_transform(audio)
        imu = self.imu_transform(imu)
        gas = self.gas_transform(gas)
        x = torch.cat((audio, imu, gas), dim=1)
        x = torch.squeeze(self.point_cnn(x), dim=1).transpose(1, 2)

        x = x + self.positional_encoding[:, :x.size(1), :].to(x.device)

        # Transformer 编码器
        x = self.transformer_encoder(x).view(x.size(0), -1)

        # 输出层
        x = self.fc_out(x)

        return x

    def _generate_positional_encoding(self, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        return pe


class SwEDModel(nn.Module):
    def __init__(self):
        super(SwEDModel, self).__init__()

        self.nb_cnn2d_filt = 64
        self.pool_size = [3, 3]
        self.rnn_size = [128, 128]
        self.fcn_size = [128, 128]

        # CNN layers
        self.conv_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()

        self.conv_layers.append(nn.Conv2d(in_channels=5, out_channels=self.nb_cnn2d_filt, kernel_size=(3, 3), padding='same'))
        self.batch_norm_layers.append(nn.BatchNorm2d(self.nb_cnn2d_filt))
        self.pooling_layers.append(nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)))
        for cnn_pool in self.pool_size:
            self.conv_layers.append(nn.Conv2d(in_channels=self.nb_cnn2d_filt, out_channels=self.nb_cnn2d_filt, kernel_size=(3, 3), padding='same'))
            self.batch_norm_layers.append(nn.BatchNorm2d(self.nb_cnn2d_filt))
            self.pooling_layers.append(nn.MaxPool2d(kernel_size=(1, cnn_pool), stride=(1, cnn_pool)))

        # RNN layers
        self.gru_layers = nn.ModuleList()
        self.gru_layers.append(nn.GRU(input_size=self.nb_cnn2d_filt, hidden_size=self.nb_cnn2d_filt, batch_first=True, bidirectional=True))
        self.gru_layers.append(nn.GRU(input_size=self.nb_cnn2d_filt * 2, hidden_size=32, batch_first=True, bidirectional=True))

        # FCN layers
        self.fcn_layers = nn.ModuleList()
        self.fcn_layers.append(nn.Linear(in_features=4096, out_features=128))
        self.fcn_layers.append(nn.Linear(in_features=128, out_features=64))

        self.bn = nn.BatchNorm1d(64)
        # Final output layer
        self.output_layer = nn.Linear(in_features=64, out_features=15)
        self.audio_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=2048,
            n_mels=64,
            hop_length=800,
        )
        self.imu_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=1000,
            n_fft=256,
            n_mels=64,
            hop_length=50,
        )
        self.gas_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=100,
            n_fft=128,
            n_mels=64,
            hop_length=5,
        )

    def forward(self, audio, imu, gas):
        audio = self.audio_transform(audio)
        imu = self.imu_transform(imu)
        gas = self.gas_transform(gas)
        x = torch.cat((audio, imu, gas), dim=1)
        # CNN
        for conv, bn, pool in zip(self.conv_layers, self.batch_norm_layers, self.pooling_layers):
            x = F.relu(bn(conv(x)))
            x = pool(x)

        # Permute and reshape for RNN input
        x = x.reshape(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1).contiguous()

        # RNN
        for gru in self.gru_layers:
            x, _ = gru(x)

        x = self.bn(x)
        x = x.reshape(x.size(0), -1)
        # FCN
        for fcn in self.fcn_layers:
            x = F.relu(fcn(x))
    
        # Output layer
        x = self.bn(x)
        x = torch.sigmoid(self.output_layer(x))
        return x
