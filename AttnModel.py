import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from copy import deepcopy


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

# class GELU(nn.Module):
#     # for older versions of PyTorch.  For new versions you can use nn.GELU() instead.
#     def __init__(self):
#         super(GELU, self).__init__()

#     def forward(self, x):
#         x = torch.nn.functional.gelu(x)
#         return x

class MRMSFFC(nn.Module):
    def  __init__(self, reduced_channel, kernel_size1, stride1, kernel_size2, stride2, in_channels=1):
        super(MRMSFFC, self).__init__()
        drate = 0.5
        self.GELU = nn.GELU()  # for older versions of PyTorch.  For new versions use nn.GELU() instead.
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
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.low_feature1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=kernel_size2, stride=stride2, bias=False, padding=stride2 - 1, dilation=1),
            nn.BatchNorm2d(64),
            self.GELU,
            nn.MaxPool2d(kernel_size=7, stride=2, padding=2),
            nn.Dropout(drate)
        )
        self.low_feature2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, bias=False, padding=2),
            nn.BatchNorm2d(128),
            self.GELU,
            nn.Conv2d(128, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(128),
            self.GELU,
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.dropout = nn.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock2D, reduced_channel, 1)

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
        x_hight = self.hight_feature1(x)
        x_low = self.low_feature1(x)
        msg = F.interpolate(x_low, size=(x_hight.size(2), x_hight.size(3)), mode='bilinear', align_corners=False)
        x_hight += msg
        x_hight = self.hight_feature2(x_hight)
        x_low = self.low_feature2(x_low)
        msg = F.interpolate(x_low, size=(x_hight.size(2), x_hight.size(3)), mode='bilinear', align_corners=False)
        x_hight += msg
        x_merge = self.dropout(x_hight)
        x_merge = self.AFR(x_merge)
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

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
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

# class LayerNorm(nn.Module):
#     "Construct a layer normalization module."

#     def __init__(self, features, eps=1e-6):
#         super(LayerNorm, self).__init__()
#         self.a_2 = nn.Parameter(torch.ones(features))
#         self.b_2 = nn.Parameter(torch.zeros(features))
#         self.eps = eps

#     def forward(self, x):
#         mean = x.mean(-1, keepdim=True)
#         std = x.std(-1, keepdim=True)
#         return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

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


class AttnSleep(nn.Module):
    def __init__(self):
        super(AttnSleep, self).__init__()

        N = 1  # number of TCE clones
        audio_model = 230
        imu_model = 200
        gas_model = 60
        d_ff = 120   # dimension of feed forward
        h = 5  # number of attention heads
        dropout = 0.1
        segment_length = 15
        reduced_channel = 30

        self.audio_mrcnn = MRCNN(reduced_channel, 3200, 40, 6400, 120)
        self.imu_mrcnn = MRCNN(reduced_channel, 200, 3, 400, 8, in_channels=3)
        self.gas_mrcnn = MRCNN(reduced_channel, 20, 1, 40, 3)
        audio_attn = MultiHeadedAttention(h, audio_model, reduced_channel)
        imu_attn = MultiHeadedAttention(h, imu_model, reduced_channel)
        gas_attn = MultiHeadedAttention(h, gas_model, reduced_channel)
        audio_ff = PositionwiseFeedForward(audio_model, d_ff, dropout)
        imu_ff = PositionwiseFeedForward(imu_model, d_ff, dropout)
        gas_ff = PositionwiseFeedForward(gas_model, d_ff, dropout)
        self.audio_tce = TCE(EncoderLayer(audio_model, deepcopy(audio_attn), deepcopy(audio_ff), reduced_channel, dropout), N)
        self.imu_tce = TCE(EncoderLayer(imu_model, deepcopy(imu_attn), deepcopy(imu_ff), reduced_channel, dropout), N)
        self.gas_tce = TCE(EncoderLayer(gas_model, deepcopy(gas_attn), deepcopy(gas_ff), reduced_channel, dropout), N)
        self.channel_attn = SELayer(reduced_channel * 3)
        self.relu = nn.ReLU()
        merge_channel = reduced_channel * 3
        x_attn = MultiHeadedAttention(h, merge_channel, audio_model)
        x_ff = PositionwiseFeedForward(merge_channel, d_ff, dropout)
        self.x_tce = TCE(EncoderLayer(merge_channel, deepcopy(x_attn), deepcopy(x_ff), audio_model, dropout), N)
        self.decoder = nn.Sequential(
            nn.Linear(audio_model * merge_channel, segment_length * reduced_channel),
            self.relu,
            nn.Dropout(dropout),
            nn.Linear(segment_length * reduced_channel, segment_length),
            nn.Sigmoid()
        )

    def forward(self, audio, imu, gas):
        audio_feat = self.audio_mrcnn(audio)
        imu_feat = self.imu_mrcnn(imu)
        gas_feat = self.gas_mrcnn(gas)
        audio_features = self.audio_tce(audio_feat)
        imu_features = self.imu_tce(imu_feat)
        gas_features = self.gas_tce(gas_feat)
        imu_features = F.interpolate(imu_features, size=(audio_features.shape[2]), mode='linear')
        gas_features = F.interpolate(gas_features, size=(audio_features.shape[2]), mode='linear')
        x_concat = torch.cat((audio_features, imu_features, gas_features), dim=1)
        x_concat = x_concat.permute(0, 2, 1)
        x_concat = self.x_tce(x_concat)
        x_concat = x_concat.permute(0, 2, 1)
        x_concat = x_concat.contiguous().view(x_concat.shape[0], -1)
        final_output = self.decoder(x_concat)
        return final_output
    
class AttnSleep2D(nn.Module):
    def __init__(self):
        super(AttnSleep2D, self).__init__()

        N = 2  # number of TCE clones
        signal_model = 870
        d_ff = 120   # dimension of feed forward
        h = 5  # number of attention heads
        dropout = 0.1
        segment_length = 15
        reduced_channel = 30

        self.mrmsffc = MRMSFFC(reduced_channel, 5, 1, 7, 3, in_channels=5)
        
        signal_attn = MultiHeadedAttention(h, signal_model, reduced_channel)
        signal_ff = PositionwiseFeedForward(signal_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(signal_model, deepcopy(signal_attn), deepcopy(signal_ff), reduced_channel, dropout), N)
        self.channel_attn = SELayer(reduced_channel)
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

######################################################################

class MRCNN_SHHS(nn.Module):
    def __init__(self, afr_reduced_cnn_size):
        super(MRCNN_SHHS, self).__init__()
        drate = 0.5
        self.GELU = nn.GELU()  # for older versions of PyTorch.  For new versions use nn.GELU() instead.
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=6, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=6, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)

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
