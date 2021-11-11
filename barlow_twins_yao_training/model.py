import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18

from pl_bolts.models.self_supervised.resnets import resnet18 as lit_ssl_resnet18


class Model(nn.Module):
    def __init__(self, feature_dim=128, dataset='cifar10'):
        super(Model, self).__init__()

        """ self.f = []
        # kiya modified this to resnet18 to make training faster.
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if dataset == 'cifar10':
                # removes the last layer classifier and the initial maxpool2d (probably since cifar images are already pretty small !)
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    self.f.append(module)
            elif dataset == 'tiny_imagenet' or dataset == 'stl10':
                if not isinstance(module, nn.Linear):
                    self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f) """
        self.f = lit_ssl_resnet18(first_conv=False, maxpool1=False, return_all_feature_maps=False)
        # projection head
        # kiya modified this for resnet 18
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)[-1]
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        #return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
        return feature, out




if __name__ == '__main__':
    from torchinfo import summary

    # the output is werird !!!!!!!!!! be careful
    lit_model = lit_ssl_resnet18(first_conv=False, maxpool1=False, return_all_feature_maps=False)
    print('lightning model')
    summary(lit_model, input_size=(1, 3, 32, 32), row_settings=("depth","var_names"), depth= 10)


    