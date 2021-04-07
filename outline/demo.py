import torch
import torchvision
import logging as log
import torchattacks

log.basicConfig(
    level=log.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        log.FileHandler("debug.log"),
        log.StreamHandler()
    ]
)

from AdvLib import Adversarisal_bench as ab

from simple_model import Net

from Get_dataset import CIFAR10_module

from Measurements import Normal_accuracy, Robust_accuracy


# get the simple model
net = Net()
path = '../model_checkpoints/simple_conv_non_robust_cifar_epoch_10.pth'
net.load_state_dict(torch.load(path))

#initialize and send the model to AdvLib
simple_model_bench = ab(net, device='cuda:2', predictor=lambda x: torch.max(x, 1)[1])

dataset = CIFAR10_module()
# prepare and setup the dataset
dataset.prepare_data()
dataset.setup()

# define  meaures
normal_acc = Normal_accuracy()
robust_acc = Robust_accuracy()

fgsm = torchattacks.FGSM(net, eps=2000/255)

attacks = [fgsm]
"""torchattacks.BIM(net, eps=8/255, alpha=2/255, steps=7), 
torchattacks.PGD(net, eps=8/255, alpha=2/255, steps=7),
torchattacks.CW(model, c=1, kappa=0, steps=1000, lr=0.01),
torchattacks.RFGSM(model, eps=8/255, alpha=4/255, steps=1), """
            

results = simple_model_bench.measure(dataset, [normal_acc, robust_acc], attacks)

log.info(results)

# TODO
# 1. use resnet-18 model (smallest resnet)
# 2. chek if attacks work (is AutoAttacks param free ?)
# 3. better print for the results
# 4. train robust model
# 5. measure dataset concentration
# 6 . ...





# # get the dataset concentration
# all_samples = None
# concent = ad.measure_dataset_concentration(all_samples)

# # get an only non-robust dataset
# train_set = None
# non_robuts_train = ad.make_non_robust_dataset(train_set)

# # make a robust dataset ?????

