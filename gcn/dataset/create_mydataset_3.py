"""
This script to create dataset and labels by clean off some NaN, do a normalization,
label smoothing and label weights by scores.

"""
import os
import pickle
import numpy as np
import pandas as pd
import torch


class_names = ["no cheat", "cheat"]
main_parts = ['LShoulder_x', 'LShoulder_y', 'RShoulder_x', 'RShoulder_y', 'LHip_x', 'LHip_y',
              'RHip_x', 'RHip_y']
main_idx_parts = [1, 2, 7, 8, -1]  # 1.5

csv_pose_file = 'E:\\kechuang\\kaochang\\Human-Falling-Detect-Tracks-master\\Data\\2.csv'
save_path = 'E:\\kechuang\\kaochang\\Human-Falling-Detect-Tracks-master\\Data\\data_train.pkl'
src_path = "E:\\kechuang\\kaochang\\Human-Falling-Detect-Tracks-master\\Data\\myDataset\\data_frames"
src_files = os.listdir(src_path)
data_frames_mod_3 = ["data_frames (23)", "data_frames (24)", "data_frames (25)", "data_frames (26)", "data_frames (27)"
                     , "data_frames (28)", "data_test"]
# Params.
smooth_labels_step = 8
n_frames = 10
skip_frame = 1

annot = pd.read_csv(csv_pose_file)

# 将主要节点缺失的一帧数据进行删除
idx = annot.iloc[:, 2:-1][main_parts].isna().sum(1) > 0
idx = np.where(idx)[0]
annot = annot.drop(idx)
# One-Hot Labels.
label_onehot = pd.get_dummies(annot['label'])
annot = annot.drop('label', axis=1).join(label_onehot)
cols = label_onehot.columns.values


def scale_pose(xy):
    """
    Normalize pose points by scale with max/min value of each pose.
    xy : (frames, parts, xy) or (parts, xy)
    """
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)
    xy_min = np.nanmin(xy, axis=1)
    xy_max = np.nanmax(xy, axis=1)
    for i in range(xy.shape[0]):
        xy[i] = ((xy[i] - xy_min[i]) / (xy_max[i] - xy_min[i])) * 2 - 1
    return xy.squeeze()


def seq_label_smoothing(labels, max_step=10):
    steps = 0
    remain_step = 0
    target_label = 0
    active_label = 0
    start_change = 0
    max_val = np.max(labels)
    min_val = np.min(labels)
    for i in range(labels.shape[0]):
        if remain_step > 0:
            if i >= start_change:
                labels[i][active_label] = max_val * remain_step / steps
                labels[i][target_label] = max_val * (steps - remain_step) / steps \
                    if max_val * (steps - remain_step) / steps else min_val
                remain_step -= 1
            continue

        diff_index = np.where(np.argmax(labels[i:i+max_step], axis=1) - np.argmax(labels[i]) != 0)[0]
        if len(diff_index) > 0:
            start_change = i + remain_step // 2
            steps = diff_index[0]
            remain_step = steps
            target_label = np.argmax(labels[i + remain_step])
            active_label = np.argmax(labels[i])
    return labels


feature_set = np.empty((0, n_frames, 14, 3))
labels_set = np.empty((0, len(cols)))
for src_file in src_files:
    div = 5
    if src_file in data_frames_mod_3:
        div = 3
    df = annot[annot['video'] == src_file].reset_index(drop=True)

    src_file_yes = os.path.join(src_path, src_file, "cheat_yes")
    src_file_no = os.path.join(src_path, src_file, "cheat_no")
    # 作弊帧
    frames_yes = os.listdir(src_file_yes)
    frames_yes = [os.path.join(src_file_yes, frame) for frame in frames_yes]

    frames_no = os.listdir(src_file_no)
    frames_no = [os.path.join(src_file_no, frame) for frame in frames_no]

    frames = frames_no + frames_yes
    frames = sorted(frames, key=lambda x: int((int(x.split("_")[-1][:-4]) / div)))
    nf = len(frames)

    # 标签平滑
    esp = 0.1
    df[cols] = df[cols] * (1 - esp) + (1 - df[cols]) * esp / (len(cols) - 1)
    df[cols] = seq_label_smoothing(df[cols].values, smooth_labels_step)

    # Separate continuous frames.
    frames = df['frame'].values
    frames_set = []
    fs = [0]
    for i in range(1, len(frames)):
        if frames[i] < frames[i-1] + 10:
            fs.append(i)
        else:
            frames_set.append(fs)
            fs = [i]
    frames_set.append(fs)

    for fs in frames_set:
        xys = df.iloc[fs, 2:-len(cols)].values.reshape(-1, 13, 3)
        # Scale pose normalize.
        xys[:, :, :2] = scale_pose(xys[:, :, :2])
        # Add center point.
        xys = np.concatenate((xys, np.expand_dims((xys[:, 1, :] + xys[:, 2, :]) / 2, 1)), axis=1)

        # Weighting main parts score.
        scr = xys[:, :, -1].copy()
        scr[:, main_idx_parts] = np.minimum(scr[:, main_idx_parts] * 1.5, 1.0)
        # Mean score.
        scr = scr.mean(1)

        # Targets.
        lb = df.iloc[fs, -len(cols):].values
        # Apply points score mean to all labels.
        lb = lb * scr[:, None]

        for i in range(xys.shape[0] - n_frames):
            feature_set = np.append(feature_set, xys[i:i+n_frames][None, ...], axis=0)
            labels_set = np.append(labels_set, lb[i:i+n_frames].mean(0)[None, ...], axis=0)


with open(save_path, 'wb') as f:
    pickle.dump((feature_set, labels_set), f)
