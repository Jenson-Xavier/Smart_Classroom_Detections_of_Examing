import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

# class MyNet(nn.modules):
#     def __init__(self):
#         super(MyNet, self).__init__()
#         self.fc1 = nn.Linear(51, 128)
#         self.fc2 = nn.Linear(128, 32)
#         self.fc3 = nn.Linear(32, 2)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         pass


class MyDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path, header=None).sample(frac=1).to_numpy()
        print(self.data.shape)
        self.x = self.data[:, 1:]
        self.y = self.data[:, 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.x[index, :], self.y[index]


model = nn.Sequential(
    nn.Linear(in_features=22, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=32),
    nn.ReLU(),
    nn.Linear(in_features=32, out_features=2),
    nn.Softmax()
)
# model.load_state_dict(torch.load('./mymodel.pth'))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

data_set = MyDataset("./tmp/pos-N.csv")
data_loader = DataLoader(dataset=data_set, batch_size=32,
                         shuffle=True, drop_last=False)

epoch = 100

# 开始训练模型
model.train()
for i in range(epoch):
    print(f'第{i+1}轮训练: ', end='')
    loss_sum = 0
    gs = 0
    ok = 0
    for data in data_loader:
        x, y = data
        x = x.to(torch.float32)
        y = y.to(torch.long)
        outputs = model(x)
        loss = loss_fn(outputs, y)
        loss_sum += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ok += (outputs.argmax(1) == y).sum()
        gs += len(y)
    print(f'loss: {loss_sum}, acc: {ok / gs}')

torch.save(model.state_dict(), "mymodel.pth")
