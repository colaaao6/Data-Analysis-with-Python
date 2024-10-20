import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 定义卷积层和全连接层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # 使用一个辅助方法来计算flatten后的输入特征数
        self.fc1 = nn.Linear(in_features=self._get_flatten_size(), out_features=100)  # 输入大小将在下面计算
        self.fc2 = nn.Linear(in_features=100, out_features=10)

    def _get_flatten_size(self):
        # 为了知道flatten之后的元素数量，我们可以让一个测试张量通过网络
        with torch.no_grad():
            x = torch.zeros((1, 3, 32, 32))  # 测试张量
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            return x.numel()

    def forward(self, x):
        # 网络前向传播
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))

        # 在传递给全连接层之前将输出展平为一维向量
        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 最后一层不使用激活函数，因为通常会使用损失函数如CrossEntropyLoss

        return x

def create_model():
    return ConvNet()
