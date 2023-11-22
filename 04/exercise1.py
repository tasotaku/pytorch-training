import torch
from torch import nn

if __name__=="__main__":
    mytensor = torch.ones(32, 3, 128, 128)
    conv = nn.Conv2d(in_channels=3, out_channels=64,  kernel_size=3, stride=2, padding=63)
    out = conv(mytensor)
    print(out.shape)

    conv2 = nn.Conv2d(in_channels=3, out_channels=256,  kernel_size=3, stride=2, padding=1)
    out2 = conv2(mytensor)
    print(out2.shape)

    conv3 = nn.Conv2d(in_channels=3, out_channels=64,  kernel_size=5, stride=2, padding=64)
    out3 = conv3(mytensor)
    print(out3.shape)

    conv4 = nn.Conv2d(in_channels=3, out_channels=256,  kernel_size=5, stride=2, padding=2)
    out4 = conv4(mytensor)
    print(out4.shape)