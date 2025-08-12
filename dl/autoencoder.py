# simple autoencoder with pytorch
import torch
import torch.nn as nn
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 64))
        self.dec = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 784))
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)

model = AE()
x = torch.randn(4, 784)
out = model(x)
print(out.shape)
