import os
import cv2
import time
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms

from copy import deepcopy
from movenet import MoveNet
from yolov5.DetectAPI import DetectAPI

save_path = 'E:\\kechuang\\kaochang\\Human-Falling-Detect-Tracks-master\\Data\\2.csv'
annot_file = "E:\\kechuang\\kaochang\\Human-Falling-Detect-Tracks-master\\Data\\1.csv"

src_path = "E:\\kechuang\\kaochang\\Human-Falling-Detect-Tracks-master\\Data\\myDataset\\data_frames"

src_files = os.listdir(src_path)
class_names = ["no cheat", "cheat"]

columns = ['video', 'frame', 'Nose_x', 'Nose_y', 'Nose_s', 'LShoulder_x', 'LShoulder_y', 'LShoulder_s',
           'RShoulder_x', 'RShoulder_y', 'RShoulder_s', 'LElbow_x', 'LElbow_y', 'LElbow_s', 'RElbow_x',
           'RElbow_y', 'RElbow_s', 'LWrist_x', 'LWrist_y', 'LWrist_s', 'RWrist_x', 'RWrist_y', 'RWrist_s',
           'LHip_x', 'LHip_y', 'LHip_s', 'RHip_x', 'RHip_y', 'RHip_s', 'LKnee_x', 'LKnee_y', 'LKnee_s',
           'RKnee_x', 'RKnee_y', 'RKnee_s', 'LAnkle_x', 'LAnkle_y', 'LAnkle_s', 'RAnkle_x', 'RAnkle_y',
           'RAnkle_s', 'label']

COCO_PAIR = [(0, 13), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (13, 7), (13, 8),  # Body
             (7, 9), (8, 10), (9, 11), (10, 12)]

POINT_COLORS = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck

LINE_COLORS = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), (77, 255, 222),
               (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77), (77, 222, 255),
               (255, 156, 127), (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]

text_colors = [(0, 255, 0), (0, 0, 255)]
class_names = ["no cheat", "cheat"]


data_frames_mod_3 = ["data_frames (23)", "data_frames (24)", "data_frames (25)", "data_frames (26)", "data_frames (27)"
                     , "data_frames (28)", "data_test"]

annot = pd.read_csv(annot_file)

detector = DetectAPI()
mnet = MoveNet()


def normalize_points_with_size(points_xy, width, height, flip=False):
    points_xy[:, 0] /= width
    points_xy[:, 1] /= height
    if flip:
        points_xy[:, 0] = 1 - points_xy[:, 0]
    return points_xy


for src_file in src_files:
    div = 5
    if src_file in data_frames_mod_3:
        div = 3
    sdf = pd.DataFrame(columns=columns)
    df = annot[annot['video'] == src_file].reset_index(drop=True)
    cur_row = 0

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

    i = 1
    while i <= nf:
        frame_path = frames[i-1]
        img = cv2.imread(frame_path)
        imgsz = img.shape[0:2][::-1]
        cls_idx = int(df[df['frame'] == i]["label"])

        bbs = np.array(detector.run(frame_path)).astype(int)
        for ib, bb in enumerate(bbs):
            xx, yy, w, h = bb
            cut_img = img[int(yy - h / 2):int(yy + h / 2), int(xx - w / 2):int(xx + w / 2), :].copy()
            bb = (xx - w / 2, yy - h / 2, xx + w / 2, yy + h / 2)
            ch, cw, _ = cut_img.shape

            keypoints = mnet.run(cut_img)[0][0]
            pos_in_img = torch.zeros(size=(17, 2))
            scores = torch.zeros(size=(17, 1))
            for ik, keypoint in enumerate(keypoints):
                x, y, confidence = keypoint
                x_pixel = int(x * ch)
                y_pixel = int(y * cw)
                pos_in_img[ik] = torch.tensor([bb[0]+y_pixel, bb[1]+x_pixel])
                scores[ik] = torch.tensor([confidence])
            pos_in_img = torch.cat([pos_in_img[0:1], pos_in_img[5:]], axis=0)
            scores = torch.cat([scores[0:1], scores[5:]], axis=0)

            pt_norm = normalize_points_with_size(pos_in_img.clone(),
                                                 imgsz[0], imgsz[1])
            pt_norm = np.concatenate((pt_norm, scores), axis=1)

            row = [src_file, i, *pt_norm.flatten().tolist(), cls_idx]
            scr = scores.mean()

            sdf.loc[cur_row] = row
            cur_row += 1

            l_pair = COCO_PAIR
            p_color = POINT_COLORS
            line_color = LINE_COLORS

            part_line = {}
            pos_in_img = torch.cat([pos_in_img, torch.unsqueeze((pos_in_img[1, :] + pos_in_img[2, :]) / 2, dim=0)])
            scores = torch.cat([scores, torch.unsqueeze((scores[1, :] + scores[2, :]) / 2, dim=0)])
            for n in range(scores.shape[0]):
                if scores[n] <= 0.05:
                    continue
                cor_x, cor_y = int(pos_in_img[n, 0]), int(pos_in_img[n, 1])
                part_line[n] = (cor_x, cor_y)
                cv2.circle(img, (cor_x, cor_y), 4, POINT_COLORS[n], -1)
            # Draw limbs
            for j, (start_p, end_p) in enumerate(l_pair):
                if start_p in part_line and end_p in part_line:
                    start_xy = part_line[start_p]
                    end_xy = part_line[end_p]
                    cv2.line(img, start_xy, end_xy, line_color[j], int((2*(scores[start_p] + scores[end_p]) + 1)[0]))

            cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), text_colors[cls_idx], 2)
            cv2.putText(img, 'Frame: {}, Pose: {}, Score: {:.4f}'.format(i, class_names[cls_idx], scr),
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_colors[cls_idx], 2)
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1

    if os.path.exists(save_path):
        sdf.to_csv(save_path, mode='a', header=False, index=False)
    else:
        sdf.to_csv(save_path, mode='w', index=False)




