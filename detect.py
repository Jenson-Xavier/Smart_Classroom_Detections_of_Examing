import os
import cv2
import time
import torch
import argparse
import numpy as np

from yolov5.DetectAPI import DetectAPI
from movenet import MoveNet

from gcn.fn import draw_single
from gcn.Track.Tracker import Detection, Tracker
from gcn.ActionsEstLoader import TSSTG

from CDM.CDM_utils import *
from checker import *

source = './gcn/dataset/myDataset/data_videos(2)/TestVideo.mp4'
TEMP_IMG_PATH = './gcn/dataset/myDataset/tmp_img.jpg'
class_names = ["no cheat", "cheat"]


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


def preproc(image):
    """preprocess function for CameraLoader.
    """
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('-C', '--camera', default=source,  # required=True,  # default=2,
                     help='Source of camera or video file path.')
    par.add_argument('--detection_input_size', type=int, default=384,
                     help='Size of input in detection model in square must be divisible by 32 (int).')
    par.add_argument('--pose_backbone', type=str, default='resnet50',
                     help='Backbone model for SPPE FastPose model.')
    par.add_argument('--show_detected', default=False, action='store_true',
                     help='Show all bounding box from detection.')
    par.add_argument('--show_skeleton', default=True, action='store_true',
                     help='Show skeleton pose.')
    par.add_argument('--save_out', type=str, default='',
                     help='Save display to video file.')
    par.add_argument('--device', type=str, default='cuda',
                     help='Device to run model on cpu or cuda.')
    par.add_argument("--n_frames", type=int, default=10,
                     help='Number of frames needed for prediction in gcn')
    par.add_argument("--weights_vote", default=[0, 0, 1.0, 0],
                     help='Weight coefficients of four prediction models, from left to right, are'
                          'cdm, cnn, dnn and gcn')
    args = par.parse_args()

    device = args.device

    inp_dets = args.detection_input_size

    # 检测模型
    detect_model = DetectAPI()
    # 骨骼点检测模型
    pose_model = MoveNet()

    # 坐标判定模型
    cdm_model = KeypointsChecker_CDM(rotor_thres=0.20, roll_thres=90, reach_thres=160)
    # 图像分类模型 cnn
    resnet_model = ResNetChecker(device=device, weights='./weights/myResNet_34_best.pt')
    # 骨骼点分类模型 dnn
    bp_model = KeypointsChecker_NN(device=device, weights='./weights/mymodel.pth')

    # Tracker.
    max_age = args.n_frames
    tracker = Tracker(max_age=max_age, n_init=3)

    weights_vote = args.weights_vote

    # gcn 时空图卷积分类
    action_model = TSSTG(weight_file="./weights/tsstg-model.pth",
                         class_names=class_names)

    resize_fn = ResizePadding(inp_dets, inp_dets)

    cam_source = args.camera
    cam = cv2.VideoCapture(cam_source)

    outvid = False
    if args.save_out != '':
        outvid = True
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(args.save_out, codec, 30, (inp_dets * 2, inp_dets * 2))

    fps_time = 0
    f = 0
    while cam.isOpened():
        f += 1
        ret, frame = cam.read()
        if not ret:
            break
        frame = resize_fn(frame)
        if f % 3 != 0:
            continue
        cv2.imwrite(TEMP_IMG_PATH, frame)
        frame = cv2.imread(TEMP_IMG_PATH)
        image = frame.copy()
        img_w, img_h = frame.shape[1], frame.shape[0]

        # Detect humans bbox in the frame with detector model.
        detected = detect_model.detect(TEMP_IMG_PATH)

        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        tracker.predict()
        # Merge two source of predicted bbox together.
        # for track in tracker.tracks:
        #     det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
        #     detected = torch.cat([detected, det], dim=0) if detected is not None else det

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

            # VISUALIZE.
            if args.show_detected:
                for bb in bbox_l:
                    frame = cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 0), 1)

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

            # 图像分类
            resnet_res = resnet_model.run(cf_cut_img).argmax() if cf_cut_img is not None else None
            # 骨骼点分类
            bp_res = bp_model.run(cf_keypoints_dnn).argmax() if cf_keypoints_dnn is not None else None
            # 坐标判定法
            cdm_res, _ = cdm_model.run(cf_keypoints_cdm) if cf_keypoints_cdm is not None else None

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'pending..'
            clr = (0, 255, 0)
            # Use 30 frames time-steps to prediction.
            gcn_res = None
            if len(track.keypoints_list) == 10:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                gcn_res = out[0].argmax()

            # 综合预测结果
            result = 0
            pred = [cdm_res, resnet_res, bp_res, gcn_res]
            for i in range(4):
                if pred[i] is not None:
                    result += weights_vote[i] * pred[i]
            result_ = round(result)

            action_name = action_model.class_names[result_]
            confidence = result if result >= 0.5 else 1 - result
            action = '{}: {:.2f}%'.format(action_name, confidence * 100)
            if action_name == 'cheat':
                clr = (0, 0, 255)
            elif action_name == 'no cheat':
                clr = (255, 200, 0)

            # VISUALIZE.
            if track.time_since_update == 0:
                if args.show_skeleton:
                    frame = draw_single(frame, track.keypoints_list[-1])
                # frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, (255, 0, 0), 2)
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr, 1)

        # Show Frame.
        frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
        frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # frame = frame[:, :, ::-1]
        fps_time = time.time()

        if outvid:
            writer.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clear resource.
    # cam.stop()
    if outvid:
        writer.release()
    cv2.destroyAllWindows()
