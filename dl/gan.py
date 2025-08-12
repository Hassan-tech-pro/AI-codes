# GAN generator and discriminator skeleton
import torch
import torch.nn as nn

class G(nn.Module):
    def __init__(self, zdim=100):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(zdim, 256), nn.ReLU(), nn.Linear(256, 784), nn.Tanh())
    def forward(self, z):
        return self.net(z)

class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 256), nn.LeakyReLU(0.2), nn.Linear(256,1), nn.Sigmoid())
    def forward(self, x):
        return self.net(x)

g = G()
d = D()
z = torch.randn(2,100)
print(g(z).shape, d(torch.randn(2,784)).shape)
