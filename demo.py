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

from outline.AdvLib import Adversarisal_bench as ab
from outline.simple_model import simple_conv_Net
from outline.Get_dataset import CIFAR10_module
from outline.Measurements import Normal_accuracy, Robust_accuracy
from outline.utils import print_dict_pretty, add_normalization_layer


# Get the model
use_simple_model = False
if use_simple_model:
    # simple model
    net = simple_conv_Net()
    path = './model_checkpoints/simple_conv_non_robust_cifar_epoch_10.pth'
    net.load_state_dict(torch.load(path))
    # normalization for inputs in [0,1] 
    model_mean = (0.5, 0.5 ,0.5)
    model_std = (0.5, 0.5 ,0.5)
else:
    # get resnet18
    from PyTorch_CIFAR10.cifar10_models.resnet import resnet18
    net = resnet18(pretrained=True)
    # normalization for inputs in [0,1]
    model_mean = (0.4914, 0.4822, 0.4465)
    model_std = (0.2471, 0.2435, 0.2616)

# add a normalization layer
net = add_normalization_layer(net, model_mean, model_std)




# make sure the data is in [0,1] ! if you use pytorch ToTensor tranform it is already taken care of.
# note we have already added a normalization layer to our models to adjust them to this data.
dataset = CIFAR10_module(mean=(0,0,0), std=(1,1,1), data_dir = "./data")
# prepare and setup the dataset
dataset.prepare_data()
dataset.setup()

# define  meaures
normal_acc = Normal_accuracy()
robust_acc = Robust_accuracy()

#initialize and send the model to AdvLib
# This has to be done before defining the attacks (and sensding the model to them) 
# otherwise the devies and the eval mode won't be set properly !!!
model_bench = ab(net, device='cuda:2', predictor=lambda x: torch.max(x, 1)[1])

fgsm = torchattacks.FGSM(net, eps=8/255)

apgd = torchattacks.APGD(net, eps=8/255, steps=10)

attacks = [fgsm]

on_train=False
measurements = [normal_acc, robust_acc]
results = model_bench.measure(dataset, measurements, attacks, on_train=on_train)

log.info(results)

print_dict_pretty(results, measurements, on_train=on_train)

# TODO
# 0. a solution for defiing the benchmark before attacks ?
# 1. check if other attacks work
# 4. train robust model
# 5. measure dataset concentration






# # get the dataset concentration
# all_samples = None
# concent = ad.measure_dataset_concentration(all_samples)

# # get an only non-robust dataset
# train_set = None
# non_robuts_train = ad.make_non_robust_dataset(train_set)

# # make a robust dataset ?????



###################################
# attacks test
""" model = net.to('cuda').eval()

atks = [torchattacks.FGSM(model, eps=8/255),
        #torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=7),
        #torchattacks.CW(model, c=1, kappa=0, steps=1000, lr=0.01),
        #torchattacks.RFGSM(model, eps=8/255, alpha=4/255, steps=1),
        #torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=7),
        #torchattacks.FFGSM(model, eps=8/255, alpha=12/255),
        #torchattacks.TPGD(model, eps=8/255, alpha=2/255, steps=7),
        #torchattacks.MIFGSM(model, eps=8/255, decay=1.0, steps=5),
        #torchattacks.APGD(model, eps=8/255, steps=10),
        #torchattacks.FAB(model, eps=8/255),
        #torchattacks.Square(model, eps=8/255),
        #torchattacks.PGDDLR(model, eps=8/255, alpha=2/255, steps=7),
        #torchattacks.DeepFool(model, steps=50, overshoot=0.02),
        #torchattacks.OnePixel(model, pixels=1, steps=75, popsize=400),
        #torchattacks.SparseFool(model, steps=20, lam=3, overshoot=0.02),
       ]
import time
from tqdm import tqdm

print("Adversarial Image & Predicted Label")

for atk in atks :
    
    print("-"*70)
    print(atk)
    
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataset.test_dataloader()):
        
        start = time.time()
        adv_images = atk(images, labels)
        labels = labels.to('cuda')
        outputs = model(adv_images)

        _, pre = torch.max(outputs.data, 1)

        total += labels.shape[0]
        correct += (pre == labels).sum()

        #imshow(torchvision.utils.make_grid(adv_images.cpu().data, normalize=True), [imagnet_data.classes[i] for i in pre])

    print('Total elapsed time (sec) : %.2f' % (time.time() - start))
    print('Robust accuracy: %.2f %%' % (100 * float(correct) / total))
    print(f'total{total}  corrrect{correct}') """


