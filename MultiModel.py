import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class ConvFeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(ConvFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.AdaptiveAvgPool2d((4, 4))
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        return x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)


class ContextualAttention(nn.Module):
    def __init__(self, input_dim):
        super(ContextualAttention, self).__init__()
        self.W = nn.Linear(input_dim, input_dim)
        self.b = nn.Parameter(torch.zeros(input_dim))
        self.uc = nn.Parameter(torch.randn(input_dim))
        
    def forward(self, x):
        u = torch.tanh(self.W(x) + self.b)
        alpha = torch.exp(torch.matmul(u, self.uc))
        alpha = alpha / alpha.sum(dim=1, keepdim=True)
        return alpha.unsqueeze(-1) * x

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 15)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(x)
        return F.sigmoid(x)

class FusionLayer(nn.Module):
    def __init__(self):
        super(FusionLayer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=96, nhead=8, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x2, x3), dim=2)
        x = self.transformer_encoder(x)
        x = x.reshape(x.size(0), -1)
        return x

class MultiModel(nn.Module):
    def __init__(self):
        super(MultiModel, self).__init__()
        self.feature_extractor_audio = ConvFeatureExtractor(1)
        self.feature_extractor_imu = ConvFeatureExtractor(3)
        self.feature_extractor_gas = ConvFeatureExtractor(1)
        
        self.attention_cough = ContextualAttention(32)
        self.attention_breath = ContextualAttention(32)
        self.attention_speech = ContextualAttention(32)
        
        self.fusion = FusionLayer()
        self.classifier = Classifier(16 * 96)
        self.audio_transform = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=64,
            melkwargs={
                "n_fft": 2048,
                "n_mels": 64,
                "hop_length": 800,
                "mel_scale": "htk",
            },
        )
        self.imu_transform = torchaudio.transforms.MFCC(
            sample_rate=1000,
            n_mfcc=64,
            melkwargs={
                "n_fft": 256,
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
        f_audio = self.feature_extractor_audio(audio)
        f_imu = self.feature_extractor_imu(imu)
        f_gas = self.feature_extractor_gas(gas)
        
        f_audio = self.attention_cough(f_audio)
        f_imu = self.attention_breath(f_imu)
        f_gas = self.attention_speech(f_gas)
        
        fused_features = self.fusion(f_audio, f_imu, f_gas)
        output = self.classifier(fused_features)
        return output
