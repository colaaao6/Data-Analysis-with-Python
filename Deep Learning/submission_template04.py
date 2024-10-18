import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Определяем слои сети
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()

        # Исправляем размер входа fc1: было 5 * 4 * 4, стало 5 * 6 * 6
        self.fc1 = nn.Linear(5 * 6 * 6, 100)  # 180 входных нейронов, 100 выходных
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # Реализуем forward pass сети
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def create_model():
    return ConvNet()


# Проверка работы модели на случайном входе
img = torch.Tensor(np.random.random((32, 3, 32, 32)))
model = create_model()
out = model(img)
print(f"Размер выхода модели: {out.shape}")

# Проверка параметров conv1
assert model.conv1.kernel_size == (5, 5), "неверный размер ядра у conv1"
assert model.conv1.in_channels == 3, "неверный размер in_channels у conv1"
assert model.conv1.out_channels == 3, "неверный размер out_channels у conv1"

# Проверка параметров pool1
assert model.pool1.kernel_size in [(2, 2), (2,)], "неверный размер ядра у pool1"

# Проверка параметров conv2
assert model.conv2.kernel_size == (3, 3), "неверный размер ядра у conv2"
assert model.conv2.in_channels == 3, "неверный размер in_channels у conv2"
assert model.conv2.out_channels == 5, "неверный размер out_channels у conv2"

# Проверка параметров pool2
assert model.pool2.kernel_size in [(2, 2), (2,)], "неверный размер ядра у pool2"

# Проверка параметров fc1
assert model.fc1.out_features == 100, "неверный размер out_features у fc1"

# Проверка параметров fc2
assert model.fc2.out_features == 10, "неверный размер out_features у fc2"

print("Все утверждения верны, модель соответствует заданным параметрам.")
