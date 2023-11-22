import torch
from torch import nn

if __name__=="__main__":
    # 問1
    _in = torch.ones(32, 1024)
    print(_in.shape)
    # 問2
    fc1 = nn.Linear(in_features=1024, out_features=256, bias=True)
    print(fc1(_in).shape)
    # 問3
    fc2 = nn.Linear(in_features=1024, out_features=2048, bias=True)
    print(fc2(_in).shape)
    # 問4
    out = fc1(_in).view(32, 16, 16)
    print(out.shape)