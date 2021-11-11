''' Compare the accuracy and change in the features of a SSL model vs standard supervised model.
'''
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks
from autoattack import AutoAttack
from art.attacks.evasion import AutoProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier



from lib.Get_dataset import CIFAR10_module
from models.SSL_linear_classifier import SSL_encoder_linear_classifier
from lib.Measurements import Normal_accuracy, Robust_accuracy, Feature_diff, Check_perturbation
from lib.AdvLib import Adversarisal_bench as ab
from lib.Attacks import AutoAttack_Wrapper, Torchattacks_Wrapper, ART_Wrapper
from lib.utils import  add_normalization_layer, print_measurement_results, save_measurements_to_csv

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


device = 'cuda:0'

def measure_model(model, model_name, dataset, feature_diff_measure):
    

    # define  meaures
    normal_acc = Normal_accuracy()
    robust_acc = Robust_accuracy()
    check_eps = Check_perturbation(8/255, 'linf')
    

    #initialize and send the model to AdvLib
    model_bench = ab(model, untrained_state_dict= None, device=device, 
                    predictor=lambda x: torch.max(x, 1)[1])

    #define attacks
    Fgsm = torchattacks.FGSM(model, eps=8/255)
    Fgsm = Torchattacks_Wrapper(Fgsm, 'Fgsm')

    Apgd_ce = AutoAttack(model, attacks_to_run = ['apgd-ce'], norm='Linf', eps=8/255, version='custom', verbose=False, device = device)
    Apgd_ce = AutoAttack_Wrapper(Apgd_ce, 'Apgd-ce')

    Apgd_dlr = AutoAttack(model, attacks_to_run = ['apgd-dlr'], norm='Linf', eps=8/255, version='custom', verbose=False, device = device)
    Apgd_dlr = AutoAttack_Wrapper(Apgd_dlr, 'Apgd-dlr')

    # untargeted attack (should I set batch_size ?) (no device available !! posted a Q in github Qs (not issues !))
    """ classifier = PyTorchClassifier(model=model, clip_values=(0.0, 1.0), loss=nn.CrossEntropyLoss(), input_shape=(1, 32, 32), nb_classes=10)
    print('classifier device: ', classifier.device)
    simple_Apgd = AutoProjectedGradientDescent(classifier, norm='inf', eps=8/255, max_iter=10, nb_random_init=1)
    simple_Apgd = ART_Wrapper(simple_Apgd, 'ART_Apgd_10_iters') """

    simple_Apgd = torchattacks.APGD(model, norm='Linf', eps = 8/255, steps=10, loss='ce')
    simple_Apgd = Torchattacks_Wrapper(simple_Apgd, 'simple_APGD_10_iters')

    attacks = [
        Fgsm, 
        #Apgd_ce,
        #Apgd_dlr,
        #simple_Apgd
            ]


    on_train=False
    on_val = False
    measurements = [normal_acc, robust_acc, feature_diff_measure] # [check_eps] 
    results = model_bench.measure_splits(dataset, measurements, attacks, on_train=on_train, on_val=on_val)
    print_measurement_results(results, measurements, on_train=on_train)
    save_measurements_to_csv(results, measurements, f'SSL_vs_supervised_results/{model_name}_results.csv', on_train=on_train)

def get_models(model_type):
    '''model_type: either
                'standard' meaning the last layer is standardly trained.  
                or 'robust_apgd' the last layer robust rained using apgd-ce eps= 8/255 linf 10 iteration torchattacks implementation
                or 'robust_pgd' to be done  
    '''
    with torch.no_grad():
        barlow_twins_encoder_path = './model_checkpoints/barlow_twins_unsupervised/0.0078125_128_128_cifar10_epoch_795.pth'
        barlow_twins = SSL_encoder_linear_classifier('barlow_twins', barlow_twins_encoder_path)

        simclr_encoder_path =   #(deleted!)'./bolt_self_supervised_training/lightning_logs_simCLR_every5th_checkpoint/version_0/checkpoints/epoch=792-step=139567.ckpt'
        simclr = SSL_encoder_linear_classifier('simCLR', simclr_encoder_path)

        # we load from the original supervised checkpoint but the SSL_encoder_linear_classifier only loads the encoder part
        supervised_path = './model_checkpoints/bolt_resnet18_supervised/best_val_acc_acc_val=88.37.ckpt'
        linear_separated_sup = SSL_encoder_linear_classifier('supervised', supervised_path)

        # get resnet18 (standard supervised)
        from huy_Supervised_models_training_CIFAR10.module import CIFAR10Module as supervised_model
        supervised = supervised_model(classifier='resnet18').load_from_checkpoint(supervised_path)
        # freeze the model
        supervised.freeze()

        standard_linear_evaluator_path = './last_layer_training_standard_logs'

        if model_type == 'standard':
            #get barlow twins (self-supervised)
            barlow_twins_linear_path = standard_linear_evaluator_path + '/barlow_twins_with_linear_layer_logs/lightning_logs/version_2/checkpoints/fixed_barlow_twins_linear_layer_trained-epoch=04-val_acc=0.884.ckpt'
            #load top linear model (lightining style!!)
            barlow_twins = barlow_twins.load_from_checkpoint(barlow_twins_linear_path, strict=False)
            barlow_twins.freeze()
            # print arch
            """ print('Barlow twins arch:')
            summary(barlow_twins, input_size=(1, 3, 32, 32), row_settings=("depth","var_names"), depth= 10)
            for name, layer_module in barlow_twins.named_modules():
                print(name)
                #print(layer_module) """

            #load simCLR
            simclr_linear_path = standard_linear_evaluator_path + '/simCLR_with_linear_layer_logs/lightning_logs/version_1/checkpoints/fixed_simCLR_linear_layer_trained-epoch=03-val_acc=0.905.ckpt'
            #load top linear model (lightining style!!)
            simclr = simclr.load_from_checkpoint(simclr_linear_path, strict=False)
            simclr.freeze()
            # print arch
            """ print()
            print('simCLR')
            summary(simclr, input_size=(1, 3, 32, 32), row_settings=("depth","var_names"), depth= 10)
            for name, layer_module in simclr.named_modules():
                print(name) """


            
            # print arch
            """ print()
            print('supervised model')
            summary(supervised, input_size=(1, 3, 32, 32), row_settings=("depth","var_names"), depth=10)
            for name, param in supervised.named_modules():
                print(name)
                #print(param)  """

            
            # get standard supervised but last layer trained separately
            supervised_linear_path = standard_linear_evaluator_path + '/supervised_with_linear_layer_logs/lightning_logs/version_1/checkpoints/fixed_supervised_linear_layer_trained-epoch=03-val_acc=0.883.ckpt'
            #load top linear model (lightining style!!)
            linear_separated_sup = linear_separated_sup.load_from_checkpoint(supervised_linear_path, strict=False)
            linear_separated_sup.freeze()

            """ for name, param in linear_separated_sup.named_modules():
                print(name) """

        elif model_type =='robust_apgd':
            def load_linear_layer(model, state_dict_path):
                model_dict = torch.load(state_dict_path) 
                bias = model_dict['1.final_linear_layer.bias']
                weights = model_dict['1.final_linear_layer.weight']
                model.final_linear_layer.weight = torch.nn.Parameter(weights)
                model.final_linear_layer.bias = torch.nn.Parameter(bias)


            # barlow
            barlow_linear_path = './Last_layer_robustly_trained_apgd_ce_eps_8_over_255_linf_torchattacks/barlow_twins_simple_apgd_linear_avdersarial_training_logs/version0/checkpoints/epoch_14.pt'
            load_linear_layer(barlow_twins, barlow_linear_path)
            barlow_twins.freeze() 

            #simCLR
            simclr_linear_path = './Last_layer_robustly_trained_apgd_ce_eps_8_over_255_linf_torchattacks/simCLR_simple_apgd_linear_avdersarial_training_logs/version0/checkpoints/epoch_10.pt'
            load_linear_layer(simclr, simclr_linear_path)
            simclr.freeze()

            # super vised last layer seperated
            # be careful there is a zombie module in huy_supervised (encoder.encoder.final_linear), I added it for bolt
            super_linear_robust_path = './Last_layer_robustly_trained_apgd_ce_eps_8_over_255_linf_torchattacks/supervised_simple_apgd_linear_avdersarial_training_logs/version1/checkpoints/epoch_4.pt'
            load_linear_layer(linear_separated_sup, super_linear_robust_path)
            linear_separated_sup.freeze()

    
    return barlow_twins, simclr, linear_separated_sup, supervised


def main():
    # current result(Values are all percentages !): 
    '''
    Linear robustly trained models:

    1. supervised:
    2021-11-02 00:17:06,403 [INFO] Test set results:
    2021-11-02 00:17:06,404 [INFO] The Normal_accuracy results:
    2021-11-02 00:17:06,404 [INFO] 85.38
    2021-11-02 00:17:06,404 [INFO] The Robust_accuracy results:
    2021-11-02 00:17:06,404 [INFO] {'simple_APGD_10_iters': {'Total_accuracy': 12.629999999999999, 'Correct_accuracy': 14.792691496837667}}
    2021-11-02 00:17:06,404 [INFO] The Feature_diff results:
    2021-11-02 00:17:06,404 [INFO] {'simple_APGD_10_iters': 8.248225212097168}
    2021-11-02 00:34:32,102 [INFO] The Robust_accuracy results:
    2021-11-02 00:34:32,102 [INFO] {'Apgd-ce': {'Total_accuracy': 2.53, 'Correct_accuracy': 2.963223237292106}}
    2021-11-02 00:34:32,102 [INFO] The Feature_diff results:
    2021-11-02 00:34:32,102 [INFO] {'Apgd-ce': 9.92147159576416}


    
    
    
    '''


    #import models
    barlow_twins, simclr, linear_separated_sup, supervised = get_models('standard') # # ('robust_apgd')

    
    models = [ barlow_twins, simclr,linear_separated_sup] # supervised,
    normlaizations = [ barlow_twins_yao, simCLR_bolts,supervised_huy,] 
    model_names = [ 'barlow_twins_yao', 'simCLR_bolts','linear_separated_supervised'] # 'supervised_huy',
    #layers to calculate the diff for (need to add 1 since we add a normalization layer)
    layer_names = [ '1.encoder.encoder.f.avgpool', '1.encoder.encoder.encoder.avgpool', '1.encoder.encoder.model.avgpool'] # '1.model.avgpool',

    #boolean array control which models to evaluate
    evaluate_flags = [ False, False, True]

    """ print('bias and weights: ')
    #print(linear_separated_sup.encoder.encoder.final_linear.bias)
    #print(linear_separated_sup.encoder.encoder.final_linear.weight)

    summary(linear_separated_sup, input_size=(1, 3, 32, 32), row_settings=("depth","var_names"), depth=10)

    for name, param in linear_separated_sup.named_modules():
        if 'linear' in name:
            print(name)
            print(param.weight)
            print(param.bias) """


    # prepare data
    # make sure the data is in [0,1] ! if you use pytorch ToTensor tranform it is already taken care of.
    # note we have already added a normalization layer to our models to adjust them to this data.
    dataset = CIFAR10_module(mean=(0,0,0), std=(1,1,1), data_dir = "./data", batch_size=512)
    dataset.prepare_data()
    dataset.setup()


    for model,normalization, layer_name, model_name, eval_model in zip(models, normlaizations, layer_names, model_names, evaluate_flags):
        if eval_model:
            print(model_name)
            # add a normalization layer to the begining 
            model = add_normalization_layer(model, normalization[0], normalization[1]) 

            # do the measurements
            measure_model(model, model_name, dataset, Feature_diff(layer_name))

        
            


if __name__ == '__main__':
    main()