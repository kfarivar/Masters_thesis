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

from lib.AdvLib import Adversarisal_bench as ab
from lib.simple_model import simple_conv_Net
from lib.Get_dataset import CIFAR10_module
from lib.Measurements import Normal_accuracy, Robust_accuracy
from lib.utils import print_measurement_results, print_train_test_val_result, add_normalization_layer
from lib.Trainer import Robust_trainer


def main(args):
    # Get the model

    if args.model=='simple':
        # simple model
        net = simple_conv_Net()
        path = './model_checkpoints/simple_conv_non_robust_cifar_epoch_10.pth'
        net.load_state_dict(torch.load(path))
        # normalization for inputs in [0,1] 
        model_mean = (0.5, 0.5 ,0.5)
        model_std = (0.5, 0.5 ,0.5)
    elif args.model=='resnet18':
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
    # This has to be done before defining the attacks (and sending the model to them) 
    # otherwise the devies and the eval mode won't be set properly !!!
    # this is weird since the attacks automatically puts the model in eval mode ?!
    model_bench = ab(net, device='cuda:2', predictor=lambda x: torch.max(x, 1)[1])


    model = net
    attacks = [torchattacks.FGSM(model, eps=8/255),
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
        ]


    if args.mode == 'measuring':
        on_train=False
        on_val = False
        measurements = [normal_acc, robust_acc]
        results = model_bench.measure_splits(dataset, measurements, attacks, on_train=on_train, on_val=on_val)
        print_measurement_results(results, measurements, on_train=on_train)

    elif args.mode == 'robust_training':
        on_val = False
        measurements = [normal_acc, robust_acc]
        #initialise optimizer 
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        loss = torch.nn.CrossEntropyLoss()
        trainer = Robust_trainer(optimizer, loss)
        results = model_bench.train_val_test(trainer, 11, dataset, measurements, attacks, train_measure_frequency=2, val_measure_frequency=2)
        print_train_test_val_result(results, measurements)
        




    


if __name__ == '__main__':
    #parse args
    import argparse
    parser = argparse.ArgumentParser()


    parser.add_argument('model', type=str, choices=['simple', 'resnet18'])

    parser.add_argument('mode', choices=['measuring', 'robust_training'])

    args = parser.parse_args()

    main(args)









