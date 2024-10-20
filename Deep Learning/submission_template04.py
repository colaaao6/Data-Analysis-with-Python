import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

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
    
    def forward(self, x):
         # Прямой проход через сеть
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))

        # Преобразование выхода в плоский вектор перед передачей в полносвязные слои
        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Последний слой без активации, так как обычно применяется функция потерь, например, CrossEntropyLoss

        return x

def create_model():
    return ConvNet()
