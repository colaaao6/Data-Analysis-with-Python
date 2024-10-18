import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Определение слоев сети
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten не является отдельным слоем, но мы можем использовать x.view() для его реализации.
        # Здесь мы рассчитываем размерность входа для fc1:
        # Исходный размер: 32x32, после conv1: 28x28, после pool1: 14x14,
        # после conv2: 12x12, после pool2: 6x6
        # Количество фильтров во втором сверточном слое - 5, поэтому:
        # 5 * 6 * 6 = 180
        self.fc1 = nn.Linear(5 * 6 * 6, 100)  # 180 входных нейронов, 100 выходных
        self.fc2 = nn.Linear(100, 10)  # 100 входных нейронов, 10 выходных

    def forward(self, x):
        # Прямой проход через сеть
        x = self.pool1(F.relu(self.conv1(x)))  # Свертка, ReLU, макс-пулинг
        x = self.pool2(F.relu(self.conv2(x)))  # Свертка, ReLU, макс-пулинг
        x = x.view(-1, 5 * 6 * 6)  # Преобразование тензора в плоский вектор
        x = F.relu(self.fc1(x))  # Полносвязный слой, ReLU
        x = self.fc2(x)  # Полносвязный слой
        return x

# Создание экземпляра модели
model = ConvNet()
print(model)
