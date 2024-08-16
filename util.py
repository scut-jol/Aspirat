import torch

# 采样率
AUDIO_SAMPLE_RATE = 16000
IMU_SAMPLE_RATE = 1000
GAS_SAMPLE_RATE = 100
BIN_NUM = 15
WIN_DARTION = 3
HOP_DURATION = 2

audio_sample_rate = AUDIO_SAMPLE_RATE
audio_win_sample = int(WIN_DARTION * AUDIO_SAMPLE_RATE)
audio_hop_sample = int(HOP_DURATION * AUDIO_SAMPLE_RATE)
imu_sample_rate = IMU_SAMPLE_RATE
imu_win_sample = int(WIN_DARTION * IMU_SAMPLE_RATE)
imu_hop_sample = int(HOP_DURATION * IMU_SAMPLE_RATE)
gas_sample_rate = GAS_SAMPLE_RATE
gas_win_sample = int(WIN_DARTION * GAS_SAMPLE_RATE)
gas_hop_sample = int(HOP_DURATION * GAS_SAMPLE_RATE)


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


def merge_silence_events(events, min_silence):
    count = 0
    while count < len(events) - 1:
        if (events[count][1] >= events[count + 1][0]) or (events[count + 1][0] - events[count][1] <= min_silence):
            events[count][1] = max(events[count + 1][1], events[count][1])
            del events[count + 1]
        else:
            count += 1


def del_min_duration_events(events, min_duration):
    count = 0
    while count < len(events) - 1:
        if events[count][1] - events[count][0] < min_duration:
            del events[count]
        else:
            count += 1
    if len(events) > 0 and events[count][1] - events[count][0] < min_duration:
        del events[count]


def Advanced_post_processing(pred, bin_duration=0.2):
    events = []
    swallow_flag = False
    for i in range(pred.shape[0]):
        pred_win = pred[i, :]
        for j in range(pred_win.shape[0]):
            bin_predicted = pred_win[j].item()
            if swallow_flag:
                if bin_predicted >= 0.4:
                    start_point = i * HOP_DURATION + j * bin_duration
                    end_point = i * HOP_DURATION + (j + 1) * bin_duration
                    events.append([start_point, end_point])
                else:
                    swallow_flag = False
            else:
                if bin_predicted >= 0.6:
                    swallow_flag = True
                    start_point = i * HOP_DURATION + j * bin_duration
                    end_point = i * HOP_DURATION + (j + 1) * bin_duration
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
