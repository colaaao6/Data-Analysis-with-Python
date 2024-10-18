import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import torchvision
from torchvision import datasets, transforms

from matplotlib import pyplot as plt
from IPython.display import clear_output
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Определение слоев сети
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten не является отдельным слоем, он будет использоваться при forward pass
        # Мы можем вычислить входной размер для первого полносвязного слоя
        # после применения всех сверток и пулинга.
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

# Пример создания экземпляра сети
net = ConvNet()
print(net)
# Эта ячейка не должна выдавать ошибку.
# Если при исполнении ячейки возникает ошибка, то в вашей реализации нейросети есть баги.
img = torch.Tensor(np.random.random((32, 3, 32, 32)))
model = ConvNet()
out = model(img)
# Создание экземпляра модели
model = ConvNet()

# Проверка параметров conv1
assert model.conv1.kernel_size == (5, 5), "Неверный размер ядра у conv1"
assert model.conv1.in_channels == 3, "Неверный размер in_channels у conv1"
assert model.conv1.out_channels == 3, "Неверный размер out_channels у conv1"

# Проверка параметров pool1
assert model.pool1.kernel_size == 2, "Неверный размер ядра у pool1"  # kernel_size может быть скалярным значением

# Проверка параметров conv2
assert model.conv2.kernel_size == (3, 3), "Неверный размер ядра у conv2"
assert model.conv2.in_channels == 3, "Неверный размер in_channels у conv2"
assert model.conv2.out_channels == 5, "Неверный размер out_channels у conv2"

# Проверка параметров pool2
assert model.pool2.kernel_size == 2, "Неверный размер ядра у pool2"  # kernel_size может быть скалярным значением

# Проверка параметров fc1
assert model.fc1.out_features == 100, "Неверный размер out_features у fc1"

# Проверка параметров fc2
assert model.fc2.out_features == 10, "Неверный размер out_features у fc2"

print("Все утверждения верны, модель соответствует заданным параметрам.")
