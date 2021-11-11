import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm








from models.models_mean_std import supervised_huy, barlow_twins_yao, simCLR_bolts


from models.simclr_module import SimCLR

from lib.utils import  add_normalization_layer, print_measurement_results

from lib.Measurements import Normal_accuracy, Robust_accuracy
import torchattacks

from autoattack import AutoAttack
from lib.Attacks import AutoAttack_Wrapper, Torchattacks_Wrapper

from lib.Get_dataset import CIFAR10_module
from lib.AdvLib import Adversarisal_bench as ab

from models.SSL_linear_classifier import SSL_encoder_linear_classifier

import logging as log
log.basicConfig(
    level=log.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        #log.FileHandler("log.log"),
        log.StreamHandler()
    ]
)

device = 'cuda:0'

def test_attack():
    '''Test a nwe attack'''
    # test for attack
    # get resnet18 (standard supervised)
    supervised_path = './huy_Supervised_models_training_CIFAR10/cifar10/resnet18/version_3/checkpoints/best_val_acc_acc_val=88.37.ckpt'
    from huy_Supervised_models_training_CIFAR10.module import CIFAR10Module as supervised_model
    supervised = supervised_model(classifier='resnet18').load_from_checkpoint(supervised_path)
    # freeze the model
    supervised.freeze() 
    
    # add a normalization layer to the begining 
    model = add_normalization_layer(supervised, supervised_huy[0], supervised_huy[1])
     

    # define  meaures
    normal_acc = Normal_accuracy()
    robust_acc = Robust_accuracy()

    #initialize and send the model to AdvLib
    model_bench = ab(model, untrained_state_dict= None, device=device, 
                    predictor=lambda x: torch.max(x, 1)[1])


    #define attacks
    Fgsm = torchattacks.FGSM(model, eps=8/255)
    Fgsm = Torchattacks_Wrapper(Fgsm, 'Fgsm')

    Apgd_ce = AutoAttack(model, attacks_to_run = ['apgd-ce'], norm='Linf', eps=8/255, version='custom') #, verbose=False)
    Apgd_ce = AutoAttack_Wrapper(Apgd_ce, 'Apgd-ce')

    Apgd_dlr = AutoAttack(model, attacks_to_run = ['apgd-dlr'], norm='Linf', eps=8/255, version='custom') #, verbose=False)
    Apgd_dlr = AutoAttack_Wrapper(Apgd_dlr, 'Apgd-dlr')

    attacks = [
        Fgsm, 
        Apgd_ce,
        Apgd_dlr
            ]

    dataset = CIFAR10_module(mean=(0,0,0), std=(1,1,1), data_dir = "./data", batch_size=512)
    dataset.prepare_data()
    dataset.setup()

    on_train=False
    on_val = False
    measurements = [normal_acc, robust_acc ]
    results = model_bench.measure_splits(dataset, measurements, attacks, on_train=on_train, on_val=on_val)
    print_measurement_results(results, measurements, on_train=on_train)
    #save_measurements_to_csv(results, measurements, f'{model_name}_results.csv', on_train=on_train)



if __name__ == '__main__': 
    with torch.no_grad():
        

        supervised_path = './huy_Supervised_models_training_CIFAR10/cifar10/resnet18/version_3/checkpoints/best_val_acc_acc_val=88.37.ckpt'
        linear_separated_sup = SSL_encoder_linear_classifier('supervised', supervised_path)


        linear_separated_sup.encoder.encoder.final_linear.weight = torch.nn.parameter.Parameter(weights)
        linear_separated_sup.encoder.encoder.final_linear.bias = torch.nn.parameter.Parameter(bias)

        print(linear_separated_sup.encoder.encoder.final_linear.bias)
        print(bias)



 




