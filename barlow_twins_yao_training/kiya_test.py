import torch
from torch import tensor 

y = tensor([1,2,3,2,1])
y_hat = tensor([1,2,5,2,6])
pred_result = (y_hat == y)

print(1/torch.sum(pred_result))

print('success')
