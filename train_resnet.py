import torch
from torchvision import models
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
from tqdm import tqdm
import os
from PIL import Image

# 相关配置参数
config = {
    'val_percent': 0.1,
    'batch_size': 32,
    'epochs': 5,
    'output_dir': './weights/',
    'save_name': 'myResNet_34.pt'
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists(config['output_dir']):
    os.mkdir(config['output_dir'])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet34(pretrained=True)
        self.fc1 = nn.Linear(in_features=1000, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=2)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.softmax(x)
        return x


class ZBCKDataset(Dataset):
    def __init__(self, img_dir, label_file):
        with open(label_file, 'r') as fp:
            content = fp.readlines()
            content = list(map(lambda t: t.split(), content))
            self.img_list = [t[0] for t in content]
            self.labels = list(map(int, [t[1] for t in content]))
        self.img_dir = img_dir
        self.labels = torch.tensor(self.labels)

    def __getitem__(self, index):
        img_path = self.img_dir + '/' + self.img_list[index]
        img_data = Image.open(img_path)
        img_data = transform(img_data)
        return img_data, self.labels[index]

    def __len__(self):
        return len(self.labels)


# 创建数据集和 dataloader
dataset = ZBCKDataset(img_dir='K:/Data/SCData',
                      label_file='K:/Data/SCData/index.txt')
val_size = int(len(dataset) * config['val_percent'])
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(
    train_dataset, batch_size=config['batch_size'], shuffle=True)
val_dataloader = DataLoader(
    val_dataset, batch_size=config['batch_size'], shuffle=True)

# 设置模型，损失函数，优化器
model = ResNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
min_val_loss = float('inf')

# 开始训练
print('================== Training Start ==================')
for i in range(config['epochs']):
    print()
    print('======== Epoch {:} / {:} ========'.format(i + 1, config['epochs']))

    train_loss = 0
    model.train()
    for batch in tqdm(train_dataloader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        loss = loss_fn(outputs, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = train_loss / len(train_dataloader)     # 计算平均误差
    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    # Validation
    print('Running Validation...')
    model.eval()
    val_accuracy = 0
    val_loss = 0
    for batch in tqdm(val_dataloader):
        x, y = batch
        # print(y)
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            outputs = model(x)
            loss = loss_fn(outputs, y)
            val_loss += loss.item()
            val_accuracy += (outputs.argmax(1) == y).sum()
            # print(outputs.argmax(1))

    val_loss = val_loss / len(val_dataloader)
    # print(val_accuracy, len(val_dataset))
    val_accuracy = val_accuracy / len(val_dataset)

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(),
                   config['save_name'].split('.')[0] + '_best.pt')

    print("  Validation Loss: {0:.2f}".format(val_loss))
    print("  Accuracy: {0:.2f}".format(val_accuracy))

print('================== Training Complete ==================')

# 保存模型
print("Saving model to %s" % config['output_dir'])
torch.save(model.state_dict(), config['save_name'])
