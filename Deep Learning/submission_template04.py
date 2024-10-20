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
        
        # 全连接层将在第一次 forward 调用时定义
        self.fc1 = None
        self.fc2 = nn.Linear(in_features=100, out_features=10)

        # 用于存储 flatten 层的输入大小
        self.flatten_size = None

    def _initialize_fc_layers(self, x):
        if self.flatten_size is None:
            with torch.no_grad():
                x = self.pool1(self.conv1(x))
                x = self.pool2(self.conv2(x))
                self.flatten_size = x.numel() // x.size(0)
                self.fc1 = nn.Linear(in_features=self.flatten_size, out_features=100).to(x.device)

    def forward(self, x):
        # 网络前向传播
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))

        # 如果 fc1 还没有初始化，则进行初始化
        self._initialize_fc_layers(x)

        # 在传递给全连接层之前将输出展平为一维向量
        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 最后一层不使用激活函数，因为通常会使用损失函数如CrossEntropyLoss

        return x

def create_model():
    return ConvNet()
