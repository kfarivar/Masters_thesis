import torch
import torch.nn as nn

# print the model structure 

from torchinfo import summary


from pl_bolts.models.self_supervised.resnets import resnet18

net = resnet18()

summary(net, input_size=(1, 3, 32, 32), row_settings=("depth","var_names"), depth= 10)