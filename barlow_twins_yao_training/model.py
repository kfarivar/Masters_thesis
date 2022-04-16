import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.models.self_supervised.resnets import resnet18 as lit_ssl_resnet18


class Model(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2048, feature_dim=2048, dataset='cifar10'):
        super(Model, self).__init__()
        # encoder
        self.f = lit_ssl_resnet18(first_conv=False, maxpool1=False, return_all_feature_maps=False)
        # projection head
        self.g = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True), 
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, feature_dim, bias=True)
            )

    def forward(self, x):
        x = self.f(x)[-1]
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return feature, out




    