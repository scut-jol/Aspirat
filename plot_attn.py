import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
attn = torch.load('attention_weights.pt')
cnn_signal = torch.load('cnn_signal_cpu.pt')
audio = torch.load("audio.pt")
imu = torch.load("imu.pt")
gas = torch.load("gas.pt")
output = torch.load("ouput.pt")
sequence = torch.load("sequence.pt")

# Parameters
batch_size, num_heads, seq_length, _ = attn.shape
d_model = 64
d_head = d_model // num_heads

# Generate ones tensor
ones_value = torch.ones(batch_size, num_heads, seq_length, d_head).float()

# Calculate context
context = torch.matmul(attn, ones_value)
context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
upsampled_tensor = F.interpolate(context.unsqueeze(1), size=(16, 61), mode='bilinear', align_corners=False)

# Colors for IMU channels
imu_colors = ['r', 'g', 'b']

# Plotting
for sample_idx in range(cnn_signal.shape[0]):
    attention_map = upsampled_tensor[sample_idx, :, :][0].numpy()
    mfcc_db = torchaudio.functional.amplitude_to_DB(cnn_signal[sample_idx, 0], multiplier=10, amin=1e-10, db_multiplier=0)

    fig, axes = plt.subplots(7, 1, figsize=(12, 24))

    # Plot MFCC heatmap
    sns.heatmap(mfcc_db, ax=axes[0], cmap='viridis')
    axes[0].invert_yaxis()
    axes[0].set_title('MFCC')

    # Plot attention map heatmap
    sns.heatmap(attention_map, ax=axes[1], cmap='viridis')
    axes[1].invert_yaxis()
    axes[1].set_title('Attention Map')

    # Plot audio data
    axes[2].plot(audio[sample_idx, 0].numpy())
    axes[2].set_title('Audio')

    # Plot IMU data (all channels in one subplot)
    for i in range(imu.shape[1]):
        axes[3].plot(imu[sample_idx, i].numpy(), color=imu_colors[i], label=f'Channel {i+1}')
    axes[3].set_title('IMU Channels')
    axes[3].legend()

    # Plot gas data
    axes[4].plot(gas[sample_idx, 0].numpy())
    axes[4].set_title('Gas')

    # Plot output data
    axes[5].plot(output[sample_idx].numpy())
    axes[5].set_title('Output')

    # Plot sequence data
    axes[6].plot(sequence[sample_idx].numpy())
    axes[6].set_title('Sequence')

    plt.tight_layout()
    plt.savefig(f'mim_picture/data_plots_sample_{sample_idx}.png')
    plt.show()
