import cv2
import torch

# 加载模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False)
model.load_state_dict(torch.load('./weights/yolov5m.pt'))

# 使用OpenCV读取图片
img = cv2.imread('path/to/your/image.jpg')

# 进行预测
results = model(img)

# 解析结果
predictions = results.pred
boxes = predictions[:, :4]  # 边界框坐标 (x1, y1, x2, y2)
scores = predictions[:, 4]  # 置信度
categories = predictions[:, 5]  # 类别

# 显示或保存结果
results.show()  # 显示图片和边界框
results.save(save_dir='results/')  # 保存结果到指定文件夹
