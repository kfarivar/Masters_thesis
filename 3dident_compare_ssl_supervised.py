from dataclasses import dataclass
from random import sample
import torch
import torchattacks
from autoattack import AutoAttack
from werkzeug import test

from models.SSL_linear_classifier import SSL_encoder_linear_classifier
from lib.Measurements import Normal_accuracy, Robust_accuracy, Feature_diff, Save_sample_images #  Check_perturbation, Calculate_mean_classifier,  Mean_classifier_accuracy
from lib.AdvLib import Adversarisal_bench as ab
from lib.Attacks import AutoAttack_Wrapper, Torchattacks_Wrapper, ART_Wrapper, Hans_wrapper
from lib.utils import  Normalize_input, print_measurement_results, save_measurements_to_csv

from lib.Get_dataset import Causal_3Dident
from huy_Supervised_models_training_CIFAR10.module import Causal3DidentModel

from torchinfo import summary


""" import logging as log
log.basicConfig(
    level=log.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        #log.FileHandler("log.log"),
        log.StreamHandler()
    ]
) """

def modify_batch(batch):
    '''for 3dident, turn regression to classification'''
    images, _, latents, index = batch
    # get classes
    labels = Causal3DidentModel.spotlight_label_from_latent(latents)

    # select the subset of data we want
    mask = labels != -1
    images = images[mask]
    labels = labels[mask]
    index = index[mask]

    return images, labels, index
    

def measure_model(model, model_name, dataset, feature_diff_measure, mode, args):
    
    # define  meaures
    normal_acc = Normal_accuracy()
    robust_acc = Robust_accuracy()
    save_samples = Save_sample_images(total=args.total_samples, path=args.image_path)
    #check_eps = Check_perturbation(8/255, 'linf')
    

    #initialize and send the model to AdvLib
    model_bench = ab(model, batch_modifier=modify_batch, device=args.device, 
                    predictor=lambda x: torch.max(x, 1)[1])

    

    #define attacks
    Fgsm = torchattacks.FGSM(model, eps=8/255)
    Fgsm = Torchattacks_Wrapper(Fgsm, 'Fgsm')

    Pgd = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10)
    Pgd = Torchattacks_Wrapper(Pgd, 'torchattacks_pgd')

    fast_fgsm = torchattacks.FFGSM(model, eps=8/255, alpha=10/255)
    fast_fgsm = Torchattacks_Wrapper(fast_fgsm, 'fast_Fgsm')

    Apgd_ce = AutoAttack(model, attacks_to_run = ['apgd-ce'], norm='Linf', eps=8/255, version='custom', verbose=True, device = args.device)
    Apgd_ce = AutoAttack_Wrapper(Apgd_ce, 'Apgd-ce')

    Apgd_dlr = AutoAttack(model, attacks_to_run = ['apgd-dlr'], norm='Linf', eps=8/255, version='custom', verbose=False, device = args.device)
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
        #simple_Apgd,

        #Apgd_dlr,
        #hans_pgd,
        #Pgd,
        #fast_fgsm,
            ]

    on_train=False
    on_val = False
    measurements = [normal_acc, robust_acc] # [check_eps]
    if feature_diff_measure != None:
        measurements +=  [feature_diff_measure]
    if args.save_samples:
        measurements += [save_samples]

    results = model_bench.measure_splits(dataset, measurements, attacks, on_train=on_train, on_val=on_val, val_shuffle=True, test_shuffle=True)
    print_measurement_results(results, measurements, on_train=on_train)
    save_measurements_to_csv(results, measurements, f'3dident_SSL_vs_supervised_results/{mode}_{model_name}_results.csv', on_train=on_train)


def get_models(model_type, get_bt=False, get_simclr = False, get_simsiam=False, get_supervised=False, get_separated=False):
    '''model_type: either
                'standard' meaning the last layer is standardly trained.  
                or 'robust_apgd' the last layer robust rained using apgd-ce eps= 8/255 linf 10 iteration torchattacks implementation
    '''

    # For the checkpoints sending the version of the linear layer should be enough (unless it is an old version, rename best as best_[rest] !)

    with torch.no_grad():

        barlow_twins = simclr = simsiam = linear_separated_sup = supervised = None

        if get_supervised:
            # get resnet18 (standard supervised)
            from huy_Supervised_models_training_CIFAR10.module import Causal3DidentModel as supervised_model
            supervised_path = './huy_Supervised_models_training_CIFAR10/3dident_logs/resnet18/version_1/checkpoints/best_val_acc_acc_val=99.24.ckpt'
            supervised = supervised_model(classifier='resnet18').load_from_checkpoint(supervised_path)
            # freeze the model
            supervised.freeze()


        standard_linear_evaluator_path = './3dident_last_layer_training_standard_logs' 

        #robust_linear_evaluator_path = './last_layer_training_robust_logs'

        if model_type == 'standard':
            mode = model_type
            if get_bt:
                """ #get barlow twins (self-supervised)
                barlow_twins_linear_path = standard_linear_evaluator_path + '/barlow_twins_with_linear_layer_logs/lightning_logs/version_2'
                #load top linear model (lightining style!!)
                barlow_twins = SSL_encoder_linear_classifier.load_from_checkpoint(barlow_twins_linear_path, mode=mode, strict=False)
                barlow_twins.freeze() """
                raise NotImplementedError()
                

            if get_simclr:
                simclr_linear_path = standard_linear_evaluator_path + '/simCLR_with_linear_layer_logs/lightning_logs/small_arch_20epochs'
                #load top linear model (lightining style!!)
                simclr = SSL_encoder_linear_classifier.load_from_checkpoint(simclr_linear_path, mode=mode, strict=False)
                simclr.freeze()
                # print arch
                
                
            if get_simsiam:
                """ simsiam_linear_path = standard_linear_evaluator_path + '/simsiam_with_linear_layer_logs/lightning_logs/version_1'
                simsiam = SSL_encoder_linear_classifier.load_from_checkpoint(simsiam_linear_path, mode=mode, strict=False)
                simsiam.freeze() """
                raise NotImplementedError()

            if get_separated:
                # get standard supervised but last layer trained separately
                supervised_linear_path = standard_linear_evaluator_path + '/supervised_with_linear_layer_logs/lightning_logs/version_0'
                #load top linear model (lightining style!!)
                linear_separated_sup = SSL_encoder_linear_classifier.load_from_checkpoint(supervised_linear_path, mode=mode, strict=False)
                linear_separated_sup.freeze()


        """ elif model_type =='robust_apgd':
            mode = 'robust'

            if get_bt:
                # barlow
                barlow_linear_path = robust_linear_evaluator_path + '/barlow_twins_simple_apgd_linear_avdersarial_training_logs/version0'
                barlow_twins = SSL_encoder_linear_classifier.load_from_checkpoint(barlow_linear_path, mode=mode)
                barlow_twins.freeze() 

            if get_simclr:
                #simCLR
                simclr_linear_path = robust_linear_evaluator_path + '/supervised_simCLR_simple_apgd_linear_avdersarial_training_logs/best_version2'
                simclr = SSL_encoder_linear_classifier.load_from_checkpoint(simclr_linear_path, mode=mode)
                simclr.freeze()

            if get_simsiam:
                simsiam_linear_path = robust_linear_evaluator_path + '/simsiam_simple_apgd_linear_avdersarial_training_logs/version0'
                simsiam = SSL_encoder_linear_classifier.load_from_checkpoint(simsiam_linear_path, mode=mode)
                simsiam.freeze()

            if get_separated:
                # supervised last layer seperated
                # be careful there is a zombie module in huy_supervised (encoder.encoder.final_linear), I added it for bolt.
                super_linear_robust_path = robust_linear_evaluator_path + '/no_augmentation_supervised_supervised_simple_apgd_linear_avdersarial_training_logs/version4'
                linear_separated_sup = SSL_encoder_linear_classifier.load_from_checkpoint(super_linear_robust_path ,mode=mode)
                linear_separated_sup.freeze() """

      
    return simclr, linear_separated_sup, supervised, barlow_twins, simsiam

def main(args):

    

    #import models
    mode =  'standard'
    simclr, linear_separated_sup, supervised, barlow_twins, simsiam = get_models(mode , get_bt=False, get_simclr = True, get_simsiam=False, 
                                                                                        get_supervised=True, get_separated=True)

                        
        
    models = [ barlow_twins, simclr, simsiam, linear_separated_sup, supervised] 
    model_names = [ 'barlow_twins_yao', 'simCLR_bolts', 'simsiam', 'linear_separated_supervised', 'standard_supervised'] 

    #layers to calculate the diff for (need to add 'model' to begining since we add a normalization layer)
    layer_names = [ 'model.encoder.encoder.f.avgpool', 'model.encoder.encoder.encoder.avgpool', 'model.encoder.encoder.online_network.encoder.avgpool', 
                    'model.encoder.encoder.model.avgpool', 'model.model.avgpool'] # 'model.model.avgpool',

    
    # prepare data
    # make sure the data is in [0,1] ! if you use pytorch ToTensor tranform it is already taken care of.
    # note we will add a normalization layer to our models to adjust them to this data.
    # since we use the dav library we activate the index for validation/testset
    dataset = Causal_3Dident(data_dir='/home/kiarash_temp/adversarial-components/3dident_causal', 
                                augment_train=False, no_normalization=True, batch_size=args.batch_size, 
                                num_workers=20, val_subset=args.val_ratio, val_include_index=True)
    dataset.setup()


    """ for model , name in zip(models, model_names):
        if model is not None:
            print()
            print(name)
            summary(model, input_size=(1, 3, 224, 224), row_settings=("depth","var_names"), depth= 10)
            #for name, layer_module in model.named_modules():
            #   print(name) """ 


    for model, layer_name, model_name in zip(models, layer_names, model_names):
        if model != None:
            print(model_name)
            # add a normalization layer to the begining 
            mean_per_channel = [0.4327, 0.2689, 0.2839]
            std_per_channel = [0.1201, 0.1457, 0.1082]
            model = Normalize_input(mean_per_channel, std_per_channel, model)
            # VERY IMPORTANT:
            # When we wrap lightining module in a pytorch Sequential applying the .to(device) will not move the lightining module !
            # I need to wrap it again 
            # I decided to use an lightining module for notmalization layer !

            
            # do the measurements
            measure_model(model, model_name, dataset, Feature_diff(layer_name), args.exp_name+mode, args )


    







if __name__ == '__main__':

    @dataclass
    class Arg:
        exp_name = '3dident_init_' # name of experiment
        val_ratio = 1
        device = 'cuda:4'
        batch_size = 128
        save_samples = False
        total_samples = 50 # num samples to save
        image_path = exp_name # path to save the sample images



    args = Arg()
    
    main(args)