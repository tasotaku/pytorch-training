import torch
from torch import nn
from models import MyModule

if __name__=="__main__":
    mymodel = MyModule()
    mytensor = torch.ones(32, 3, 128, 128)
    out = mymodel(mytensor)
    print(out.shape)