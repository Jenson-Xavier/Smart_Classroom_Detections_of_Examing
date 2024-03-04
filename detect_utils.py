import os
os.environ["TFHUB_CACHE_DIR"] = "./movenet"
import cv2
import time
import torch
import argparse
import numpy as np

from yolov5.DetectAPI import DetectAPI

from gcn.fn import draw_single
from gcn.Track.Tracker import Detection, Tracker
from gcn.ActionsEstLoader import TSSTG
from collections import OrderedDict

from CDM.CDM_utils import *
from checker import *
from movenet import MoveNet

frame_num = 0       # 视频总帧数
total_time = 0.0    # 视频总时长
div = 3             # 每多少帧处理一次

def ResizePadding(height, width):
    desized_size = (height, width)

    def resizePadding(image, **kwargs):
        old_size = image.shape[:2]
        max_size_idx = old_size.index(max(old_size))
        ratio = float(desized_size[max_size_idx]) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        if new_size > desized_size:
            min_size_idx = old_size.index(min(old_size))
            ratio = float(desized_size[min_size_idx]) / min(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])

        image = cv2.resize(image, (new_size[1], new_size[0]))
        delta_w = desized_size[1] - new_size[1]
        delta_h = desized_size[0] - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
        return image
    return resizePadding


# def preproc(image):
#     """preprocess function for CameraLoader.
#     """
#     image = resize_fn(image)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     return image


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


def run(use_cdm, use_bp, use_resnet, use_gcn, video_path):
    vid_name = video_path.split("/")[-1][:-4]
    TEMP_IMG_PATH = './gcn/dataset/myDataset/tmp_img.jpg'
    class_names = ["no cheat", "cheat"]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    inp_dets = 384
    max_age = 10
    weights_vote = [0.1, 0.15, 0.25, 0.5]

    use = [int(use_cdm), int(use_resnet), int(use_bp), int(use_gcn)]
    s = sum([weights_vote[i] * use[i] for i in range(4)])
    weights_vote = [weights_vote[i] / s for i in range(4)]

    fix = ' '.join(map(str, use))
    save_vid_wk = f"./video/{vid_name}_{fix}_1.mp4"
    save_vid_wtk = f"./video/{vid_name}_{fix}_0.mp4"

    # 检测模型
    detect_model = DetectAPI(device=device)
    # 骨骼点检测模型
    pose_model = MoveNet()
    # 坐标判定模型
    cdm_model = KeypointsChecker_CDM(rotor_thres=0.20, roll_thres=90, reach_thres=160)
    # 图像分类模型 cnn
    resnet_model = ResNetChecker(device=device, weights='./weights/myResNet_34_best.pt')
    # 骨骼点分类模型 dnn
    bp_model = KeypointsChecker_NN(device=device, weights='./weights/mymodel.pth')
    # gcn 时空图卷积分类
    action_model = TSSTG(weight_file="./weights/tsstg-model.pth",
                         class_names=class_names)
    resize_fn = ResizePadding(inp_dets, inp_dets)

    tracker = Tracker(max_age=max_age, n_init=3)

    cap = cv2.VideoCapture(video_path)

    frame_rate = cap.get(5)
    frame_num = cap.get(7)
    total_time = frame_num / frame_rate    # min

    codec = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(save_vid_wk, codec, int(frame_rate / div), frameSize=(inp_dets * 2, inp_dets * 2))
    writer_n = cv2.VideoWriter(save_vid_wtk, codec, int(frame_rate / div), frameSize=(inp_dets * 2, inp_dets * 2))

    fps_time = 0
    f = 0
    cheat_ts = []
    while cap.isOpened():
        f += 1
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize_fn(frame)
        if f % div != 0:
            # writer.write(frame)
            # writer_n.write(frame)
            continue
        cv2.imwrite(TEMP_IMG_PATH, frame)
        frame = cv2.imread(TEMP_IMG_PATH)
        img_w, img_h = frame.shape[1], frame.shape[0]

        # Detect humans bbox in the frame with detector model.
        detected = detect_model.detect(TEMP_IMG_PATH)

        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        tracker.predict()

        frame_n = frame.copy()
        detections = []  # List of Detections object for tracking.
        if detected is not None:
            bbs = np.array(detected[:, :4]).astype(int)
            confs = np.array(detected[:, 4])
            # Predict skeleton pose of each bboxs.
            keypoints_l, bbox_scores_l, bbox_l, keypoints2_l, keypoints3_l, cut_img_l = [], [], [], [], [], []
            for ib, (bb, conf) in enumerate(zip(bbs, confs)):
                xx, yy, w, h = bb
                x1, y1, x2, y2 = max(int(xx - w / 2), 0), max(int(yy - h / 2), 0), min(int(xx + w / 2), img_w - 1), min(
                    int(yy + h / 2), img_h - 1)
                cut_img = frame[y1: y2, x1: x2, :].copy()
                bb = torch.tensor((xx - w / 2, yy - h / 2, xx + w / 2, yy + h / 2))
                ch, cw, _ = cut_img.shape

                keypoints = pose_model.run(cut_img)[0][0]

                pos_in_img_init = torch.zeros(size=(17, 2))
                scores = torch.zeros(size=(17, 1))
                for ik, keypoint in enumerate(keypoints):
                    x, y, confidence = keypoint
                    x_pixel = int(x * ch)
                    y_pixel = int(y * cw)
                    pos_in_img_init[ik] = torch.tensor([bb[0] + y_pixel, bb[1] + x_pixel])
                    scores[ik] = torch.tensor([confidence])

                pos_in_img = torch.cat([pos_in_img_init[0:1], pos_in_img_init[5:]], axis=0)
                scores = torch.cat([scores[0:1], scores[5:]], axis=0)

                bbox_l.append(bb)
                bbox_scores_l.append(torch.tensor(conf))
                keypoints_l.append(torch.cat([pos_in_img, scores], dim=1))
                keypoints2_l.append(torch.tensor(pos_in_img_init))
                keypoints3_l.append(torch.tensor(keypoints))
                cut_img_l.append(cut_img)

            # Create Detections object.
            detections = [Detection(kpt2bbox(keypoints_l[ips][:, :2].numpy()),
                                    keypoints_l[ips].numpy(),
                                    keypoints_l[ips][:, 2].mean().numpy(),
                                    keypoints2_l[ips],
                                    keypoints3_l[ips],
                                    cut_img_l[ips]) for ips in range(len(bbox_l))]

        # Update tracks by matching each track information of current and previous frame or
        # create a new track if no matched.
        tracker.update(detections)

        # Predict Actions of each track.
        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed():
                continue

            cf_keypoints_cdm = track.cf_keypoints_cdm
            cf_keypoints_dnn = track.cf_keypoints_dnn
            cf_cut_img = track.cf_cut_img

            resnet_res, bp_res, cdm_res = None, None, None
            # 图像分类
            if use_resnet:
                resnet_res = resnet_model.run(cf_cut_img).argmax() if cf_cut_img is not None else None
            # 骨骼点分类
            if use_bp:
                bp_res = bp_model.run(cf_keypoints_dnn).argmax() if cf_keypoints_dnn is not None else None
            # 坐标判定法
            if use_cdm:
                cdm_res, _ = cdm_model.run(cf_keypoints_cdm) if cf_keypoints_cdm is not None else None

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            # Use 30 frames time-steps to prediction.
            gcn_res = None
            if len(track.keypoints_list) == 10:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                gcn_res = out[0].argmax()

            # 综合预测结果
            result = 0
            pred = [cdm_res, resnet_res, bp_res, gcn_res]
            nu = 0
            for j in range(4):
                if pred[j] is not None and use[j]:
                    result += weights_vote[j] * pred[j]
                    nu += 1
            if nu != 0:
                result_ = round(result)
                action_name = action_model.class_names[result_]
                confidence = result if result >= 0.5 else 1 - result
                action = '{}: {:.2f}%'.format(action_name, confidence * 100)
                if action_name == 'cheat':
                    clr = (0, 0, 255)
                elif action_name == 'no cheat':
                    clr = (255, 200, 0)

                # 若为作弊，则将作弊时刻进行记录
                if result_ == 1:
                    cur_time = f / frame_rate
                    ch = int(cur_time // 3600)
                    cm = int(cur_time % 3600 // 60)
                    cs = int(cur_time % 60)
                    # cheat_ts.append(cur_time)
                    cheat_ts.append(f"{ch}:{cm}:{cs}")
            else:
                action_name = 'pending..'
                clr = (0, 255, 0)
                action = '{}%'.format(action_name)

            # VISUALIZE.
            if track.time_since_update == 0:
                frame = draw_single(frame, track.keypoints_list[-1])
                # frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, (255, 0, 0), 2)
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr, 1)
                frame_n = cv2.putText(frame_n, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, (255, 0, 0), 2)
                frame_n = cv2.putText(frame_n, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr, 1)

        # Show Frame.
        frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
        frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        frame_n = cv2.resize(frame_n, (0, 0), fx=2., fy=2.)
        frame_n = cv2.putText(frame_n, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # frame = frame[:, :, ::-1]
        fps_time = time.time()

        writer.write(frame)
        writer_n.write(frame_n)
        cv2.imshow('frame', frame_n)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cheat_ts = list(OrderedDict.fromkeys(cheat_ts))
    writer.release()
    cv2.destroyAllWindows()
    return cheat_ts, save_vid_wk, save_vid_wtk


# if __name__ == "__main__":
#     cheat_ts, save_vid_wk, save_vid_wtk = run(True, True, False, True,
#         './gcn/dataset/myDataset/data_videos/data_video (28).mp4')
#     for t in cheat_ts:
#         print(f"{t}时刻, 疑似有人作弊")
#     print(save_vid_wk)
#     print(save_vid_wtk)
