''' Compare the accuracy and change in the features of a SSL model vs standard supervised model.
'''
from argparse import ArgumentParser
from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks
from autoattack import AutoAttack
import numpy as np


from lib.Get_dataset import CIFAR10_module
from models.SSL_linear_classifier import SSL_encoder_linear_classifier
from lib.Measurements import Normal_accuracy, Robust_accuracy, Feature_diff, Check_perturbation, Calculate_mean_classifier,  Mean_classifier_accuracy
from lib.AdvLib import Adversarisal_bench as ab
from lib.Attacks import AutoAttack_Wrapper, Torchattacks_Wrapper, ART_Wrapper, Hans_wrapper
from lib.utils import  Normalize_input, print_measurement_results, save_measurements_to_csv

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

def calculate_mean_classifier_accuracies(encoder, dataset):
    #initialize and send the model to AdvLib
    model_bench = ab(encoder, untrained_state_dict= None, device=device, 
                    predictor=lambda x: torch.max(x, 1)[1])
    # Get the mean classifier only on train !
    # Currently the train is augmented so I take the average of 20 runs to get a better estimate (like the linear layer training)
    print("calcling mean features")
    reps = 5
    cluster_centers = torch.zeros(10, 512, device=device)
    for i in range(reps):
        mean_features =  Calculate_mean_classifier(class_num=10, feature_dim=512)
        model_bench.measure_splits(dataset, [mean_features], [], on_train=True, on_val=False, on_test=False)
        cluster_centers += mean_features.mean_features
    cluster_centers = cluster_centers / reps

    print(cluster_centers)

    # calc accuracy
    mean_acc =  Mean_classifier_accuracy(cluster_centers)
    result = model_bench.measure_splits(dataset, [mean_acc], [], on_train=False, on_val=False, on_test=True)

    print_measurement_results(result, [mean_acc], on_train=False, on_val=False, on_test=True)


def measure_model(model, model_name, dataset, feature_diff_measure, mode):
    

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

    Pgd = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10)
    Pgd = Torchattacks_Wrapper(Pgd, 'torchattacks_pgd')

    fast_fgsm = torchattacks.FFGSM(model, eps=8/255, alpha=10/255)
    fast_fgsm = Torchattacks_Wrapper(fast_fgsm, 'fast_Fgsm')

    Apgd_ce = AutoAttack(model, attacks_to_run = ['apgd-ce'], norm='Linf', eps=8/255, version='custom', verbose=False, device = device)
    Apgd_ce = AutoAttack_Wrapper(Apgd_ce, 'Apgd-ce')

    Apgd_dlr = AutoAttack(model, attacks_to_run = ['apgd-dlr'], norm='Linf', eps=8/255, version='custom', verbose=False, device = device)
    Apgd_dlr = AutoAttack_Wrapper(Apgd_dlr, 'Apgd-dlr')

    """ The problem with ART is that they need their own model class and it disectes the model so training this model might not change that one !! also their Docs are shit !
    # untargeted attack (should I set batch_size ?) (no device available !! posted a Q in github Qs (not issues !))
    classifier = PyTorchClassifier(model=model, device_type='gpu', clip_values=(0.0, 1.0), loss=nn.CrossEntropyLoss(), input_shape=(3, 32, 32), nb_classes=10)
    print('classifier device: ', classifier.device)
    simple_Apgd = AutoProjectedGradientDescent(classifier, norm='inf', eps=8/255, max_iter=10, nb_random_init=1)
    art_simple_Apgd = ART_Wrapper(simple_Apgd, 'ART_Apgd_10_iters')  """

    # Celever hans lib
    hans_pgd = Hans_wrapper(model, 'hans_pgd', eps=8/255, step_size=2/255, iters=10)
    

    simple_Apgd = torchattacks.APGD(model, norm='Linf', eps = 8/255, steps=10, loss='ce')
    simple_Apgd = Torchattacks_Wrapper(simple_Apgd, 'simple_APGD_10_iters')

    attacks = [
        Fgsm, 
        Apgd_ce,
        simple_Apgd,

        #Apgd_dlr,
        #hans_pgd,
        #Pgd,
        #fast_fgsm,
            ]

    on_train=False
    on_val = False
    if feature_diff_measure != None:
        measurements = [normal_acc, robust_acc, feature_diff_measure] # [check_eps]
    else:
        measurements = [normal_acc, robust_acc]

    results = model_bench.measure_splits(dataset, measurements, attacks, on_train=on_train, on_val=on_val)
    print_measurement_results(results, measurements, on_train=on_train)
    results = save_measurements_to_csv(results, measurements, f'cifar10_SSL_vs_supervised_results/{mode}_{model_name}_results.csv', on_train=on_train)

    return results


def get_models(model_type, get_bt=False, get_simclr = False, get_simsiam=False, get_supervised=False, get_separated=False,
                sup_path=None, lin_seperated_path = None, simclr_path=None    ):
    '''model_type: either
                'standard' meaning the last layer is standardly trained.  
                or 'robust_apgd' the last layer robust rained using apgd-ce eps= 8/255 linf 10 iteration torchattacks implementation
                or 'robust_pgd' to be done  
    '''

    # For the checkpoints sending the version of the linear layer should be enough (unless it is an old version, rename best as best_[rest] !)

    with torch.no_grad():

        barlow_twins = simclr = simsiam = linear_separated_sup = supervised = None

        if get_supervised:
            # get resnet18 (standard supervised)
            from huy_Supervised_models_training_CIFAR10.module import CIFAR10Module as supervised_model
            supervised_path = sup_path
            supervised = supervised_model(classifier='resnet18').load_from_checkpoint(supervised_path)
            # freeze the model
            supervised.freeze()


        standard_linear_evaluator_path = './last_layer_training_standard_logs' # './last_layer_training_standard_logs'

        robust_linear_evaluator_path = './last_layer_training_robust_logs'

        if model_type == 'standard':
            mode = model_type
            if get_bt:
                #get barlow twins (self-supervised)
                barlow_twins_linear_path = standard_linear_evaluator_path + '/barlow_twins_with_linear_layer_logs/lightning_logs/version_2'
                #load top linear model (lightining style!!)
                barlow_twins = SSL_encoder_linear_classifier.load_from_checkpoint(barlow_twins_linear_path, mode=mode, strict=False)
                barlow_twins.freeze()
                

            if get_simclr:
                #load simCLR
                # 400epochs
                simclr_linear_path = simclr_path #standard_linear_evaluator_path + '/simCLR_with_linear_layer_logs/lightning_logs/supervised_simclr'
                #load top linear model (lightining style!!)
                simclr = SSL_encoder_linear_classifier.load_from_checkpoint(simclr_linear_path, mode=mode, strict=False)
                simclr.freeze()
                # print arch
                '''print()
                print('simCLR')
                summary(simclr, input_size=(1, 3, 32, 32), row_settings=("depth","var_names"), depth= 10)
                for name, layer_module in simclr.named_modules():
                    print(name) '''
                
            if get_simsiam:
                simsiam_linear_path = standard_linear_evaluator_path + '/simsiam_with_linear_layer_logs/lightning_logs/version_1'
                simsiam = SSL_encoder_linear_classifier.load_from_checkpoint(simsiam_linear_path, mode=mode, strict=False)
                simsiam.freeze()

            if get_separated:
                # get standard supervised but last layer trained separately
                supervised_linear_path = lin_seperated_path #standard_linear_evaluator_path + '/supervised_with_linear_layer_logs/lightning_logs/no_augmentation'
                #load top linear model (lightining style!!)
                linear_separated_sup = SSL_encoder_linear_classifier.load_from_checkpoint(supervised_linear_path, mode=mode, strict=False)
                linear_separated_sup.freeze()


        elif model_type =='robust_apgd':
            mode = 'robust'

            if get_bt:
                # barlow
                barlow_linear_path = robust_linear_evaluator_path + '/barlow_twins_simple_apgd_linear_avdersarial_training_logs/version0'
                barlow_twins = SSL_encoder_linear_classifier.load_from_checkpoint(barlow_linear_path, mode=mode)
                barlow_twins.freeze() 

            if get_simclr:
                #simCLR
                simclr_linear_path = simclr_path #robust_linear_evaluator_path + '/supervised_simCLR_simple_apgd_linear_avdersarial_training_logs/best_version2'
                simclr = SSL_encoder_linear_classifier.load_from_checkpoint(simclr_linear_path, mode=mode)
                simclr.freeze()

            if get_simsiam:
                simsiam_linear_path = robust_linear_evaluator_path + '/simsiam_simple_apgd_linear_avdersarial_training_logs/version0'
                simsiam = SSL_encoder_linear_classifier.load_from_checkpoint(simsiam_linear_path, mode=mode)
                simsiam.freeze()

            if get_separated:
                # supervised last layer seperated
                # be careful there is a zombie module in huy_supervised (encoder.encoder.final_linear), I added it for bolt.
                super_linear_robust_path = lin_seperated_path #robust_linear_evaluator_path + '/no_augmentation_supervised_supervised_simple_apgd_linear_avdersarial_training_logs/version4'
                linear_separated_sup = SSL_encoder_linear_classifier.load_from_checkpoint(super_linear_robust_path ,mode=mode)
                linear_separated_sup.freeze()

      
    return barlow_twins, simclr, simsiam, linear_separated_sup, supervised






def main():
    # current result(Values are all percentages !): 

    sup_paths_std = [
                    './huy_Supervised_models_training_CIFAR10/cifar10_logs/resnet18/version_0/checkpoints/best_val_acc_acc_val=95.52.ckpt',
                    './huy_Supervised_models_training_CIFAR10/cifar10_logs/resnet18/version_1/checkpoints/best_val_acc_acc_val=95.74.ckpt',
                    './huy_Supervised_models_training_CIFAR10/cifar10_logs/resnet18/version_2/checkpoints/best_val_acc_acc_val=95.51.ckpt'
    ]

    simclr_paths_std = ['./cifar10_last_layer_training_standard_logs/simCLR_with_linear_layer_logs/lightning_logs/version_0',
                        './cifar10_last_layer_training_standard_logs/simCLR_with_linear_layer_logs/lightning_logs/version_1',
                        './cifar10_last_layer_training_standard_logs/simCLR_with_linear_layer_logs/lightning_logs/version_2'
                    ]

    simclr_paths_adv = [
                        './last_layer_training_robust_logs/3runs_600epochs_simclr_simCLR_simple_apgd_linear_avdersarial_training_logs/version0',
                        './last_layer_training_robust_logs/3runs_600epochs_simclr_simCLR_simple_apgd_linear_avdersarial_training_logs/version1',
                        './last_layer_training_robust_logs/3runs_600epochs_simclr_simCLR_simple_apgd_linear_avdersarial_training_logs/version2',
    ]

    lin_sep_std = [
                    './cifar10_last_layer_training_standard_logs/supervised_with_linear_layer_logs/lightning_logs/version_0',
                    './cifar10_last_layer_training_standard_logs/supervised_with_linear_layer_logs/lightning_logs/version_1',
                    './cifar10_last_layer_training_standard_logs/supervised_with_linear_layer_logs/lightning_logs/version_2',
                    ]

    lin_sep_adv = [
                    './last_layer_training_robust_logs/correct_arch_supervised_supervised_simple_apgd_linear_avdersarial_training_logs/version0',
                    './last_layer_training_robust_logs/correct_arch_supervised_supervised_simple_apgd_linear_avdersarial_training_logs/version1',
                    './last_layer_training_robust_logs/correct_arch_supervised_supervised_simple_apgd_linear_avdersarial_training_logs/version2',
    ]

    exp_name = 'Sup_correct_arch_' Set this !

    for sup_path, simclr_path in zip(sup_paths_std, ):

        #import models
        mode = 'robust_apgd'   #'robust_apgd'  # 'standard'
        barlow_twins, simclr, simsiam, linear_separated_sup, supervised = get_models(mode , 
                                                                            get_bt=False, get_simclr = True, get_simsiam=False, 
                                                                            get_supervised=False, get_separated=False, sup_path=sup_path, simclr_path=simclr_path) # # ('standard')

        
        
        models = [ barlow_twins, simclr, simsiam, linear_separated_sup, supervised] # supervised,
        normlaizations = [ barlow_twins_yao, simCLR_bolts, simCLR_bolts, supervised_huy, supervised_huy] 
        model_names = [ 'barlow_twins_yao', 'simCLR_bolts', 'simsiam', 'linear_separated_supervised', 'standard_supervised'] # 'supervised_huy',
        #layers to calculate the diff for (need to add 'model' to begining since we add a normalization layer)
        layer_names = [ 'model.encoder.encoder.f.avgpool', 'model.encoder.encoder.encoder.avgpool', 'model.encoder.encoder.online_network.encoder.avgpool', 
                        'model.encoder.encoder.model.avgpool', 'model.model.avgpool'] # 'model.model.avgpool',

        


        # prepare data
        # make sure the data is in [0,1] ! if you use pytorch ToTensor tranform it is already taken care of.
        # note we will add a normalization layer to our models to adjust them to this data.
        dataset = CIFAR10_module(mean=(0,0,0), std=(1,1,1), data_dir = "./data", batch_size=512)
        dataset.prepare_data()
        dataset.setup()

        for model , name in zip(models, model_names):
            if model is not None:
                print()
                print(name)
                summary(model, input_size=(1, 3, 32, 32), row_settings=("depth","var_names"), depth= 10)
                """ for name, layer_module in model.named_modules():
                    print(name) """


        #get the mean classifier results ()
        """ for model,normalization, layer_name, model_name in zip(models, normlaizations, layer_names, model_names):
            if model != None:
                print(model_name)
                # add a normalization layer to the begining 
                if isinstance(model, SSL_encoder_linear_classifier):
                    model = model.encoder
                else:
                    print('encoder extraction not supported. remmember to use the SSL version of supervised for encoder !')
                
                encoder = Normalize_input(normalization[0], normalization[1], model)

                calculate_mean_classifier_accuracies(encoder, dataset)  """


        """ for model,normalization, layer_name, model_name in zip(models, normlaizations, layer_names, model_names):
            if model != None:
                print(model_name)
                # add a normalization layer to the begining 
                model = Normalize_input(normalization[0], normalization[1], model)

                # VERY IMPORTANT:
                # When we wrap lightining module in a pytorch Sequential applying the .to(device) will not move the lightining module !
                # I need to wrap it again 
                # I decided to use an lightining module for notmalization layer !
                

                # do the measurements
                measure_model(model, model_name, dataset, Feature_diff(layer_name), exp_name+mode ) """


    
 


if __name__ == '__main__':
    main()










""" 
# timing attacks test on simCLR:
import time
times= []
for i in range(5):
    start_time = time.time()
    measure_model(simclr, 'simclr', dataset, None)
    times.append(time.time() - start_time)


print(f"mean time --- {np.mean(times)} seconds ---" )
print(times) """

""" 
Logs: (5 runs)

Torachattacks APGD 10 iters:
mean time --- 8.834980869293213 seconds ---
[12.411853075027466, 7.9803149700164795, 7.916016578674316, 7.950958251953125, 7.915761470794678]

Torchattacks Pgd 10 iters:
mean time --- 8.880766105651855 seconds ---
[12.299943685531616, 8.107594966888428, 7.894123554229736, 7.93968939781189, 8.162478923797607]

Clever Hans Pgd 10 iters:
mean time --- 8.935820722579956 seconds ---
[12.398164987564087, 8.253995418548584, 8.098965883255005, 8.004812717437744, 7.92316460609436]

Obviosuly fast_fgsm is much faster ~ 3 sec 

"""        