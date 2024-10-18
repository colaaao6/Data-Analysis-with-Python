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
        
        # 全连接层
        # 计算fc1的输入尺寸：(初始宽度/4) * (初始高度/4) * conv2的滤波器数量
        # 因为初始尺寸是32x32，经过两次池化后变成8x8
        # 所以fc1的输入尺寸是 8*8*5 = 320
        self.fc1 = nn.Linear(320, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # 网络前向传播
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # 将张量展平以便传递给全连接层
        x = x.view(-1, 320)  # 或者使用 x.flatten(1) 替换 x.view(-1, 320)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 如果使用交叉熵损失函数，则最后一层通常不使用激活函数
        
        return x

def create_model():
    return ConvNet()

# 创建模型实例
model = create_model()

# 检查模型架构
# conv1
assert model.conv1.kernel_size == (5, 5), "conv1的内核大小不正确"
assert model.conv1.in_channels == 3, "conv1的in_channels不正确"
assert model.conv1.out_channels == 3, "conv1的out_channels不正确"

# pool1
assert model.pool1.kernel_size == 2 or model.pool1.kernel_size == (2, 2), "pool1的内核大小不正确"

# conv2
assert model.conv2.kernel_size == (3, 3), "conv2的内核大小不正确"
assert model.conv2.in_channels == 3, "conv2的in_channels不正确"
assert model.conv2.out_channels == 5, "conv2的out_channels不正确"

# pool2
assert model.pool2.kernel_size == 2 or model.pool2.kernel_size == (2, 2), "pool2的内核大小不正确"

# fc1
assert model.fc1.out_features == 100, "fc1的输出特征数不正确"
# fc2
assert model.fc2.out_features == 10, "fc2的输出特征数不正确"

# 如果所有检查都通过了，打印成功信息
print("所有检查都通过了！")
