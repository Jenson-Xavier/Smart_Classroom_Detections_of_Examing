# from movenet import MoveNet
import numpy as np
from torchvision import transforms, models
import cv2
from PIL import Image
import torch.nn as nn
import torch
import time


class Checker:
    def __init__(self, a=1.0, b=0.5, c=1.0, device='cpu', resnet_weights='./weights/myResNet_best.pt', keypoints_weights='./weights/mymodel.pth') -> None:
        self.a = a
        self.b = b
        self.c = c
        self.resnet_checker = ResNetChecker(device=device, weights=resnet_weights)
        self.keypoints_checker = KeypointsChecker(device=device, weights=keypoints_weights)

    def run(self, img):
        #tp = time.time()
        pred1 = self.resnet_checker.run(img)
        #print(f'Resnet: {time.time() - tp}')
        #tp = time.time()
        pred2, pred3 = self.keypoints_checker.run(img)
        #print(f'Movenet: {time.time() - tp}')
        pred1 *= self.a
        pred2 *= self.b
        pred3 *= self.c
        return np.argmax(pred1 + pred2 + pred3)


class ResNetChecker:
    def __init__(self, device='cpu', weights='./weights/myResNet_best.pt') -> None:
        self.device = device
        self.model = ResNet().to(device)
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.model.load_state_dict(torch.load(weights))
        self.model.to(self.device)
        self.model.eval()

    def run(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img.astype('uint8'))
        img = self.transform(img).unsqueeze(0)
        img = img.to(self.device)
        result = self.model(img)
        result = result.squeeze().detach().cpu().numpy()
        return result


class KeypointsChecker:
    def __init__(self, device='cpu', weights='./weights/mymodel.pth') -> None:
        self.checker_nn = KeypointsChecker_NN(device=device, weights=weights)
        self.checker_hm = KeypointsChecker_HM()
        self.mvnet = MoveNet()

    def run(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        keypoints = self.mvnet.run(img)

        pred1 = self.checker_nn.run(keypoints)
        pred2 = self.checker_hm.run(keypoints)

        return pred1, pred2


class KeypointsChecker_NN:
    def __init__(self, device='cpu', weights='./weights/mymodel.pth') -> None:
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(in_features=22, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2),
            nn.Softmax()
        )
        self.model.load_state_dict(torch.load(weights))
        self.model.to(device)
        self.model.eval()

    def run(self, keypoints):
        keypoints = np.array(keypoints).flatten(order='C')
        keypoints = keypoints[[0,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22,24,25,27,28,30,31]]
        keypoints = torch.tensor(keypoints).to(self.device)
        pred = self.model(keypoints)
        return pred.detach().cpu().numpy()


class KeypointsChecker_HM:
    def __init__(self) -> None:
        pass

    def run(self, keypoints):
        return np.array([0., 0.])


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet34(pretrained=True)
        self.fc1 = nn.Linear(in_features=1000, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
