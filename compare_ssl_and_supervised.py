''' Compare the accuracy and change in the features of a SSL model vs standard supervised model.
'''
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks


from lib.Get_dataset import CIFAR10_module
from models.SSL_linear_classifier import SSL_encoder_linear_classifier
from PyTorch_CIFAR10.cifar10_models.resnet import resnet18
from lib.Measurements import Normal_accuracy, Robust_accuracy, Feature_diff
from lib.AdvLib import Adversarisal_bench as ab
from lib.utils import  add_normalization_layer, print_measurement_results

from models.models_mean_std import supervised_huy, barlow_twins_yao, simCLR_bolts

import logging as log
log.basicConfig(
    level=log.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        #log.FileHandler("log.log"),
        log.StreamHandler()
    ]
)

from torchinfo import summary


device = 'cuda:1'

def measure_model(model, dataset, feature_diff_measure):
    # define  meaures
    normal_acc = Normal_accuracy()
    robust_acc = Robust_accuracy()
    

    #initialize and send the model to AdvLib
    model_bench = ab(model, untrained_state_dict= None, device=device, 
                    predictor=lambda x: torch.max(x, 1)[1])

    attacks = [torchattacks.FGSM(model, eps=8/255), 
            #torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=7),
            #torchattacks.APGD(model, eps=8/255, steps=7), # default norm inf
            ]


    on_train=False
    on_val = False
    measurements = [normal_acc, robust_acc, feature_diff_measure]
    results = model_bench.measure_splits(dataset, measurements, attacks, on_train=on_train, on_val=on_val)
    print_measurement_results(results, measurements, on_train=on_train)



def main():
    #import models
    
    #get barlow twins (self-supervised)
    barlow_twins_encoder_path = './model_checkpoints/barlow_twins_unsupervised/0.0078125_128_128_cifar10_model_1000.pth'
    barlow_twins_linear_path = './barlow_twins_with_linear_layer_logs/lightning_logs/version_5/checkpoints/fixed_barlow_twins_linear_layer_trained-epoch=14-val_acc=0.885.ckpt'
    barlow_twins = SSL_encoder_linear_classifier('barlow_twins', barlow_twins_encoder_path)

    """ print('BEFOR LOAD')
    for name, param in barlow_twins.named_parameters():
        if 'final_linear_layer' in name :
            print()
            print(name)
            print(param) """


    # only loades the linear part
    """ print()
    print('File values:')
    print( torch.load(barlow_twins_linear_path)['state_dict'] )
 """

    # some fuckery to get load to work !
    linear_states =  torch.load(barlow_twins_linear_path)['state_dict'] 
    linear_states['weight'] = linear_states.pop('final_linear_layer.weight')
    linear_states['bias'] = linear_states.pop('final_linear_layer.bias')
    barlow_twins.final_linear_layer.load_state_dict(linear_states)


    #barlow_twins.load_from_checkpoint(barlow_twins_linear_path, strict=False)
    barlow_twins.freeze()

    """ print('AFTER LOAD')
    for name, param in barlow_twins.named_parameters():
        #if 'final_linear_layer' in name:
            print()
            print(name)
            print(param) """

    """ summary(barlow_twins, input_size=(1, 3, 32, 32), row_settings=("depth","var_names"), depth= 5)
    for name, layer_module in barlow_twins.named_modules():
        if 'pre_process' in name:
            print(name)
            #print(layer_module) """

    # get resnet18 (standard supervised)
    supervised = resnet18(pretrained=True)
    # freeze the model
    supervised.requires_grad_(False)
    supervised.eval()

    """ print()
    print('supervised model')
    summary(supervised, input_size=(1, 3, 32, 32), row_settings=("depth","var_names"), depth=4)
    for name, param in supervised.named_modules():
        if 'avgpool' in name:
            print(name)
            #print(param) """


    
    """ #simCLR
    simclr_encoder_path = '/home/farivar/adversarial-components/model_checkpoints/simCLR_unsupervised/epoch=285-step=50335.ckpt'
    simclr = SSL_encoder_linear_classifier('simCLR', simclr_encoder_path)
    print()
    print('simCLR')
    summary(simclr, input_size=(1, 3, 32, 32), row_settings=("depth","var_names"), depth= 7) """

        



    

    models = [supervised, barlow_twins]
    normlaizations = [supervised_huy, barlow_twins_yao]
    #layers to calculate the diff for (need to add 1 since we add a normalization layer)
    layer_names = ['1.avgpool', '1.encoder.pre_process.0']
    #layer_names = ['output', 'output']


    # prepare data
    # make sure the data is in [0,1] ! if you use pytorch ToTensor tranform it is already taken care of.
    # note we have already added a normalization layer to our models to adjust them to this data.
    dataset = CIFAR10_module(mean=(0,0,0), std=(1,1,1), data_dir = "./data", batch_size=512)
    dataset.prepare_data()
    dataset.setup()


    for model,normalization, layer_name in zip(models, normlaizations, layer_names):
        model_name = type(model).__name__
        print()
        print(model_name)
        if (model_name == 'SSL_encoder_linear_classifier'):
            print(model.encoder.model)

        # add a normalization layer to the begining 
        model = add_normalization_layer(model, normalization[0], normalization[1]) 

        # do the measurements
        measure_model(model, dataset, Feature_diff(layer_name)) 

        
            


if __name__ == '__main__':
    main()