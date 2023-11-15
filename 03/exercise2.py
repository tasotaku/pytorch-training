import torch
from torch.nn import Module

class MyModule(Module):

    def __init__(self, mytensor: torch.Tensor, elem_add: int, elem_multiply: int):
        super().__init__()
        self.mytensor = mytensor
        self.elem_add = elem_add
        self.elem_multiply = elem_multiply

    def forward(self, x: torch.Tensor):
        ans2 = x + self.mytensor
        ans3 = ans2 + self.elem_add
        ans4 = ans3 * self.elem_multiply

        return ans2, ans3, ans4

if __name__=="__main__":
    mymodel = MyModule(torch.ones((3, 3)), 4, 6)
    
    p2out, p3out, p4out = mymodel(torch.full((3, 3), 2))

    print("===== problem 2 =====")
    print(repr(p2out))
    print("===== problem 3 =====")
    print(repr(p3out))
    print("===== problem 4 =====")
    print(repr(p4out))           
