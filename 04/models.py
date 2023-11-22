import torch
from torch import nn

class MyModule(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=256,  kernel_size=5, stride=8)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=65536, out_features=64, bias=True)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        print(x.shape)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(32, -1)
        print(x.shape)
        x = self.linear(x)

        return x