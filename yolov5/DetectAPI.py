import os
import sys
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (Profile, check_img_size, non_max_suppression, scale_boxes, xyxy2xywh)
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from utils.general import check_img_size
import numpy as np
import cv2

class DetectAPI:
    def __init__(self,
                 weights='./weights/yolov5m6.pt',  # model path or triton URL
                 data='./data/coco128.yaml',  # dataset.yaml path
                 conf_thres=0.5,  # confidence threshold
                 iou_thres=0.45,  # NMS IOU threshold
                 max_det=100,  # maximum detections per image
                 device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 classes=0,  # filter by class: --class 0, or --class 0 2 3
                 agnostic_nms=False,  # class-agnostic NMS
                 augment=False,  # augmented inference
                 visualize=False,  # visualize features
                 half=False,  # use FP16 half-precision inference
                 dnn=False,  # use OpenCV DNN for ONNX inference
                 vid_stride=1,  # video frame-rate stride
                 ):
        self.vid_stride = vid_stride
        self.augment = augment
        self.visualize = visualize
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        self.classes = classes

        # Load model
        device = select_device(device)
        self.model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
    
    def run(self, img0, imgsz=(1280, 1280)):
        imgsz = check_img_size(imgsz, self.stride)
        img = letterbox(img0, imgsz, stride=self.stride, auto=self.pt)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255
        if len(img.shape) == 3:
            img = img[None]

        pred = self.model(img, augment=self.augment, visualize=self.visualize)

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes,
                                   self.agnostic_nms, self.max_det)
        
        # Process predictions
        fd = []
        for i, det in enumerate(pred):
            im0 = img0.copy()
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                    fd.append(xywh)
        return fd

    def detect(self,
            source='0',  # file/dir/URL/glob/screen/0(webcam)
            imgsz=(1280, 1280),  # inference size (height, width)
            ):
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        webcam = source.isnumeric() or source.endswith('.streams')
        screenshot = source.lower().startswith('screen')

        imgsz = check_img_size(imgsz, s=self.stride)

        # Dataloader
        bs = 1  # batch_size
        if webcam:
            dataset = LoadStreams(source, img_size=imgsz, stride=self.stride, auto=self.pt, vid_stride=self.vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=self.stride, auto=self.pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=self.stride, auto=self.pt, vid_stride=self.vid_stride)

        # Run inference
        results = []
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                pred = self.model(im, augment=self.augment, visualize=self.visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                           max_det=self.max_det)

            # Process predictions
            fd = []
            for i, det in enumerate(pred):
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                        fd.append(torch.tensor(xywh + [float(conf), float(cls), 0.0]))
            if len(fd) != 0:
                results.append(torch.cat(fd).view(-1, 7))
        if len(results) == 1:
            return results[0]
        return results


if __name__ == '__main__':
    detector = DetectAPI(weights='./weights/yolov5m6.pt')
    #print(detector.run('./data/T01.jpg'))
    img = cv2.imread('./data/T01.jpg')
    print(detector.run(img))
    # rt = detector.run(r"C:\Users\Hodaka Chen\Pictures\T01.mp4")
    # for i in rt:
    #     print(i)
    #     print('============================================================')
