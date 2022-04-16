import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


from models.models_mean_std import supervised_huy, barlow_twins_yao, simCLR_bolts



from lib.utils import   print_measurement_results

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



def print_model_arch():
    from torchinfo import summary
    
    print('3dident model:')
    from torchvision import models
    base_encoder_class = {
        "rn18": models.resnet18,
        "rn50": models.resnet50,
        "rn101": models.resnet101,
        "rn152": models.resnet152,
    }['rn18']
    n_latents = 10
    encoder = base_encoder_class(False, num_classes=n_latents * 10)
    projection = torch.nn.Sequential(*[torch.nn.LeakyReLU(), torch.nn.Linear(n_latents * 10, 8 + 1)])
    f = torch.nn.Sequential(*[encoder, projection])
    summary(f, input_size=(1, 3, 224, 224), row_settings=("depth","var_names"), depth= 10) 

    """ print('Bolts model:')
    from pl_bolts.models.self_supervised.resnets import resnet18
    summary(resnet18(maxpool1=False), input_size=(1, 3, 32 , 32), row_settings=("depth","var_names"), depth= 10)
    """
    #first_conv=False, maxpool1=False, return_all_feature_maps=False

    """ from barlow_twins_yao_training.barlowtwins_module import BarlowTwins

    barlow_linear_path = '/home/kiarash_temp/adversarial-components/barlow_twins_yao_training/barlow_twins_resnet18_logs_and_chekpoints/lightning_logs/without_cosine_decay/checkpoints/epoch=398-best_val_loss_val_loss=264.6981506347656.ckpt'
    barlow_twins = BarlowTwins.load_from_checkpoint(barlow_linear_path, strict=False)
    summary(barlow_twins, input_size=(1, 3, 32 , 32), row_settings=("depth","var_names"), depth= 10)  """


    """ for name, layer_module in simclr.named_modules():
        print(name) """

    print('simclr')
    from bolt_self_supervised_training.simclr.simclr_module import SimCLR
    path = '/home/kiarash_temp/adversarial-components/bolt_self_supervised_training/simclr/3dident_simCLR_resnet18_logs_and_chekpoints/lightning_logs/version_0/checkpoints/epoch=0_best_val_loss=7.298301696777344_online_val_acc=0.41.ckpt'
    model = SimCLR.load_from_checkpoint(path, strict=False)
    summary(model, input_size=(1, 3, 224 , 224), row_settings=("depth","var_names"), depth= 10)


    from huy_Supervised_models_training_CIFAR10.module import Causal3DidentModel as supervised_model
    supervised_path = './huy_Supervised_models_training_CIFAR10/3dident_logs/resnet18/version_1/checkpoints/best_val_acc_acc_val=99.24.ckpt'
    supervised = supervised_model(classifier='resnet18').load_from_checkpoint(supervised_path)
    summary(supervised, input_size=(1, 3, 224 , 224), row_settings=("depth","var_names"), depth= 10)



if __name__ == '__main__': 

    print_model_arch()

    





 




