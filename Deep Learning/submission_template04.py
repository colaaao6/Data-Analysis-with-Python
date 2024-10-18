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
        
        # Flatten layer не требуется явно создавать, так как его функциональность
        # можно реализовать с помощью метода .view() или .flatten()
        
        # Полносвязные слои
        # Для расчета входного размера fc1 нужно знать выходной размер после conv2 и maxpool2
        # После второго пулинга размер будет (ширина/4) * (высота/4) * количество_фильтров_conv2
        # Так как начальный размер 32x32, то после двух пулингов он станет 8x8
        # Поэтому входной размер для fc1 будет 8*8*5 = 320
        self.fc1 = nn.Linear(320, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # Прямой проход через сеть
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # Преобразование тензора перед подачей на полносвязный слой
        x = x.view(-1, 320)  # Или используйте x.flatten(1) вместо x.view(-1, 320)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Последний слой обычно без активации, если используется кросс-энтропия
        
        return x
        # 创建模型实例
model = ConvNet()

# conv1
assert model.conv1.kernel_size == (5, 5), "conv1 的内核大小不正确"
assert model.conv1.in_channels == 3, "conv1 的输入通道数不正确"
assert model.conv1.out_channels == 3, "conv1 的输出通道数不正确"

# pool1
assert model.pool1.kernel_size == 2 or model.pool1.kernel_size == (2, 2), "pool1 的内核大小不正确"

# conv2
assert model.conv2.kernel_size == (3, 3), "conv2 的内核大小不正确"
assert model.conv2.in_channels == 3, "conv2 的输入通道数不正确"
assert model.conv2.out_channels == 5, "conv2 的输出通道数不正确"

# pool2
assert model.pool2.kernel_size == 2 or model.pool2.kernel_size == (2, 2), "pool2 的内核大小不正确"

# fc1
assert model.fc1.out_features == 100, "fc1 的输出特征数不正确"
# fc2
assert model.fc2.out_features == 10, "fc2 的输出特征数不正确"

print("所有断言都通过了，网络结构正确。")
