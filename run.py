from yolov5.DetectAPI import DetectAPI
import cv2
import numpy as np
from PIL import Image
from checker import Checker
import torch
from config import *
import argparse
import time


device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source',
                    type=str,
                    default='./data/videos/Test.mp4')
parser.add_argument('-o', '--output',
                    type=str,
                    default='./runs/output.mp4')
parser.add_argument('-d', '--device',
                    type=str,
                    default=device)
parser.add_argument('--show',
                    action='store_false')
args = parser.parse_args()

device = args.device
source = args.source
output = args.output
show = args.show

detector = DetectAPI(weights=Yolo_weights, data='./data/coco128.yaml')
checker = Checker(
    a=Pred_weight[0],
    b=Pred_weight[1],
    c=Pred_weight[2],
    device=device,
    resnet_weights=Resnet_weights,
    keypoints_weights=Keypoints_weights
)

print('Start Processig ...')
t1 = time.time()

cap = cv2.VideoCapture(source)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output, fourcc, Target_fps, frame_size)
compress_ration = Target_fps / fps

prior = -1
now = 0
stu_pos = None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if int(now * compress_ration) == prior:
        now += 1
        continue
    else:
        now += 1
        prior += 1

    if prior % 2 == 0:
        stu_pos = detector.run(frame)
        
    for i, det in enumerate(stu_pos):
        x, y, w, h = det
        place = (int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2))
        cut_img = frame[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2), :].copy()
        
        #tp1 = time.time()
        result = checker.run(cut_img)
        #print('T2', time.time() - tp1)

        if result == 1:
            cv2.rectangle(frame, (place[0], place[1]), (place[2], place[3]), (0, 0, 255), 5)
        else:
            cv2.rectangle(frame, (place[0], place[1]), (place[2], place[3]), (0, 255, 0), 5)

    video_writer.write(frame)

    if show:
        cv2.imshow('Test', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    
    # print('----------')
    # if now > 20:
    #     break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

t2 = time.time()
print('Finish.')
print(f'Time Cost: {t2 - t1}s')
                 