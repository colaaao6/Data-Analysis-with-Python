import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import torchvision
from torchvision import datasets, transforms

from matplotlib import pyplot as plt
from IPython.display import clear_output
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Определение сверточных и полносвязных слоев
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=self._get_flatten_size(), out_features=100)  # размер входа будет вычислен ниже
        self.fc2 = nn.Linear(in_features=100, out_features=10)

    def _get_flatten_size(self):
        # Чтобы узнать количество элементов после flatten, можно пропустить через сеть тестовый тензор
        with torch.no_grad():
            x = torch.zeros((1, 3, 32, 32))  # Тестовый тензор
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            return x.numel()

    def forward(self, x):
        # Прямой проход через сеть
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))

        # Преобразование выхода в плоский вектор перед передачей в полносвязные слои
        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Последний слой без активации, так как обычно применяется функция потерь, например, CrossEntropyLoss

        return x
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
