import torch
import numpy as np

data = np.array([
        [[85, 78], [67, 82], [92, 88], [75, 70], [60, 64]],
        [[70, 68], [77, 72], [85, 90], [60, 65], [78, 76]],
        [[80, 84], [88, 87], [66, 68], [72, 73], [64, 60]]
    ])

tensor_data = torch.tensor(data, dtype=float)
print("===== problem 1 =====")
#print(tensor_data)
print(repr(tensor_data.size()))

permuted_tensor = torch.permute(tensor_data, (2, 0, 1))
print("===== problem 2 =====")
print(repr(permuted_tensor))
print(repr(permuted_tensor.size()))

permuted_tensor3 = permuted_tensor.sum(dim=0)
print("===== problem 3 =====")
print(repr(permuted_tensor3))

permuted_tensor4 = permuted_tensor3.mean(dim=1)
print("===== problem 4 =====")
print(permuted_tensor4)

sum_class = permuted_tensor3.sum(dim=1)
num_class_member = tensor_data.size(dim=1)
ans = sum_class / num_class_member
print("===== problem 5 =====")
print(ans)

