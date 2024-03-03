import os
import cv2
import time
import torch
import argparse
import numpy as np

from yolov5.DetectAPI import DetectAPI
from movenet import MoveNet

from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

source = './dataset/myDataset/data_videos(2)/data_video (21).mp4'
TEMP_IMG_PATH = './dataset/myDataset/tmp_img.jpg'
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
    par.add_argument("--n_frames", type=int, default=10)
    args = par.parse_args()

    device = args.device

    # DETECTION MODEL.
    inp_dets = args.detection_input_size
    detect_model = DetectAPI()

    pose_model = MoveNet()

    # Tracker.
    max_age = args.n_frames
    tracker = Tracker(max_age=max_age, n_init=3)

    # Actions Estimate.
    action_model = TSSTG(weight_file="E:\\kechuang\\kaochang\\Human-Falling-Detect-Tracks-master\\Actionsrecognition\\saved\\TSSTG(pts+mot)-01(cf+hm-hm)_1\\tsstg-model.pth",
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
            keypoints_l = []
            bbox_scores_l = []
            bbox_l = []
            for ib, (bb, conf) in enumerate(zip(bbs, confs)):
                xx, yy, w, h = bb
                cut_img = frame[max(int(yy - h / 2), 0):min(int(yy + h / 2), int(frame.shape[1]) - 1), max(int(xx - w / 2), 0):min(int(xx + w / 2), int(frame.shape[0]) - 1), :].copy()
                bb = torch.tensor((xx - w / 2, yy - h / 2, xx + w / 2, yy + h / 2))
                ch, cw, _ = cut_img.shape

                keypoints = pose_model.run(cut_img)[0][0]
                pos_in_img = torch.zeros(size=(17, 2))
                scores = torch.zeros(size=(17, 1))
                for ik, keypoint in enumerate(keypoints):
                    x, y, confidence = keypoint
                    x_pixel = int(x * ch)
                    y_pixel = int(y * cw)
                    pos_in_img[ik] = torch.tensor([bb[0] + y_pixel, bb[1] + x_pixel])
                    scores[ik] = torch.tensor([confidence])
                pos_in_img = torch.cat([pos_in_img[0:1], pos_in_img[5:]], axis=0)
                scores = torch.cat([scores[0:1], scores[5:]], axis=0)

                bbox_l.append(bb)
                bbox_scores_l.append(torch.tensor(conf))
                keypoints_l.append(torch.cat([pos_in_img, scores], dim=1))


            # Create Detections object.
            detections = [Detection(kpt2bbox(keypoints_l[ips][:, :2].numpy()),
                                    keypoints_l[ips].numpy(),
                                    keypoints_l[ips][:, 2].mean().numpy()) for ips in range(len(bbox_l))]

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

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'pending..'
            clr = (0, 255, 0)
            # Use 30 frames time-steps to prediction.
            if len(track.keypoints_list) == 10:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                action_name = action_model.class_names[out[0].argmax()]
                action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
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
