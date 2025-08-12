# simple cnn with pytorch on random data
import torch
import torch.nn as nn
import torch.optim as optim

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10)
        )
    def forward(self, x):
        return self.net(x)

model = SmallCNN()
x = torch.randn(8,3,32,32)
y = model(x)
print('out', y.shape)
