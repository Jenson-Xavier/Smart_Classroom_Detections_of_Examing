import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

src_path = "E:\\kechuang\\kaochang\\Human-Falling-Detect-Tracks-master\\Data\\myDataset\\data_frames"
src_files = os.listdir(src_path)
save_path = "E:\\kechuang\\kaochang\\Human-Falling-Detect-Tracks-master\\Data\\1.csv"

data_frames_mod_3 = ["data_frames (23)", "data_frames (24)", "data_frames (25)", "data_frames (26)", "data_frames (27)"
                     , "data_frames (28)", "data_test"]

class_names = ["no cheat", "cheat"]
colors = [(0, 255, 0), (0, 0, 255)]

cols = ["video", "frame", "label"]
all_frames = []
df = pd.DataFrame(columns=cols)
for src_file in src_files:
    div = 5
    if src_file in data_frames_mod_3:
        div = 3
    src_file_yes = os.path.join(src_path, src_file, "cheat_yes")
    src_file_no = os.path.join(src_path, src_file, "cheat_no")
    # 作弊帧
    frames_yes = os.listdir(src_file_yes)
    frames_yes = [os.path.join(src_file_yes, frame) for frame in frames_yes]
    idx_yes = [int((int(frame.split("_")[-1][:-4]) / div)) for frame in frames_yes]  # 提取数字，作为作弊帧的索引

    frames_no = os.listdir(src_file_no)
    frames_no = [os.path.join(src_file_no, frame) for frame in frames_no]
    idx_no = [int((int(frame.split("_")[-1][:-4]) / div)) for frame in frames_no]

    frames = frames_no + frames_yes
    frames = sorted(frames, key=lambda x: int((int(x.split("_")[-1][:-4]) / div)))
    nf = len(frames)

    video = np.array([src_file] * nf)
    frame = np.arange(1, nf+1)
    label = np.array([0] * nf)
    label[idx_yes] = 1
    rows = np.stack([video, frame, label], axis=1)
    df = df._append(pd.DataFrame(rows, columns=cols),
                    ignore_index=True)
    all_frames.append(frames)
df.to_csv(save_path, index=False)

choose = 24
data_frame = src_files[choose]
imgs = [cv2.imread(p) for p in all_frames[choose]]
ni = len(imgs)
i = 0
df = df[df['video'] == data_frame].reset_index(drop=True)
while True:
    frame = imgs[i]
    cls_name = class_names[int(df.iloc[i, -1])]
    color = colors[int(df.iloc[i, -1])]
    frame = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)
    frame = cv2.putText(frame, 'Frame: {} Pose: {}'.format(i+1, cls_name),
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow('frame', frame)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        i += 1
        if i >= ni:
            break
        continue
    elif key == ord('a'):
        if i >= 1:
            i -= 1
        continue
    else:
        break




