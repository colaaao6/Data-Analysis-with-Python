import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 定义卷积层和池化层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # 初始化全连接层的输入大小
        self._initialize_fc_layers()

    def _initialize_fc_layers(self):
        # 使用一个临时张量来计算flatten后的输入大小
        with torch.no_grad():
            x = torch.zeros((1, 3, 32, 32))  # 测试张量
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            flatten_size = x.numel() // x.size(0)

        # 根据计算出的flatten大小定义全连接层
        self.fc1 = nn.Linear(in_features=flatten_size, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=10)

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
    # 创建模型实例
model = create_model()

# 检查参数
assert model.conv1.kernel_size == (5, 5), "Неверный размер ядра у conv1"
assert model.conv1.in_channels == 3, "Неверный размер in_channels у conv1"
assert model.conv1.out_channels == 3, "Неверный размер out_channels у conv1"

# Проверка параметров pool1
assert model.pool1.kernel_size == 2 or model.pool1.kernel_size == (2, 2), "Неверный размер ядра у pool1"

# Проверка параметров conv2
assert model.conv2.kernel_size == (3, 3), "Неверный размер ядра у conv2"
assert model.conv2.in_channels == 3, "Неверный размер in_channels у conv2"
assert model.conv2.out_channels == 5, "Неверный размер out_channels у conv2"

# Проверка параметров pool2
assert model.pool2.kernel_size == 2 or model.pool2.kernel_size == (2, 2), "Неверный размер ядра у pool2"

# Проверка параметров fc1
assert model.fc1.out_features == 100, "Неверный размер out_features у fc1"

# Проверка параметров fc2
assert model.fc2.out_features == 10, "Неверный размер out_features у fc2"

print("Все проверки пройдены успешно!")

# 进行一次前向传递以验证网络是否可以正常运行
img = torch.Tensor(np.random.random((32, 3, 32, 32)))
out = model(img)
print("模型前向传递成功！")
