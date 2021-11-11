from argparse import ArgumentParser
import json
import torch
from torch import nn
from torch.nn import functional as F
import torchattacks
from autoattack import AutoAttack

from lib.AdvLib import Adversarisal_bench as ab
from models.models_mean_std import supervised_huy, barlow_twins_yao, simCLR_bolts
from models.SSL_linear_classifier import SSL_encoder_linear_classifier
from lib.utils import  print_train_test_val_result, add_normalization_layer, save_training_results_to_csv
from lib.Get_dataset import CIFAR10_module
from lib.Measurements import Normal_accuracy, Robust_accuracy, Feature_diff, Loss_measure
from lib.Attacks import AutoAttack_Wrapper, Torchattacks_Wrapper
from lib.Trainer import Robust_trainer

import logging as log
log.basicConfig(
    level=log.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        log.FileHandler("log.log"),
        log.StreamHandler()
    ]
)


def determin_version(root):
    import glob
    version_folders = glob.glob(root+"/version*")

    import re
    def get_trailing_number(s):
        m = re.search(r'\d+$', s)
        return int(m.group()) if m else None

    if not version_folders:
        return 0
    else:
        versions = [get_trailing_number(path) for path in version_folders ]
        return max(versions)+1



def main(args):
    device = 'cuda:'+ str(args.device)

    #create save paths
    root_save_dir= f'./last_layer_training_robust_logs/{args.model}_{args.attack}_linear_avdersarial_training_logs'
    version = determin_version(root_save_dir)
    save_dir = root_save_dir + f'/version{version}'
    model_save_path = save_dir + '/checkpoints'
    tb_logs_path = save_dir + '/tb_logs'
    from pathlib import Path
    Path(save_dir).mkdir(parents=True, exist_ok=False)
    Path(model_save_path).mkdir(parents=True, exist_ok=False)
    Path(tb_logs_path).mkdir(parents=True, exist_ok=False)

    print('Verion is:', version)

    # save hyper params (https://stackoverflow.com/questions/42318915/saving-python-argparse-file)
    with open(save_dir+'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # dataset specific modifications
    if args.model == 'barlow_twins':
        # normalization used in yao's barlow twins (the stds are off I should invetigate and retrain the model if necessary!)
        mean=barlow_twins_yao[0]
        std=barlow_twins_yao[1]
        feature_layer_name = '1.encoder.encoder.f.avgpool'
    
    elif args.model == 'simCLR':
        # taken from pl_bolts.transforms.dataset_normalizations.cifar10_normalization
        mean= simCLR_bolts[0]
        std= simCLR_bolts[1]
        feature_layer_name ='1.encoder.encoder.encoder.avgpool'

    elif args.model =='supervised':
        mean = supervised_huy[0]
        std = supervised_huy[1]
        feature_layer_name ='1.encoder.encoder.model.avgpool'

    model = SSL_encoder_linear_classifier(args.model, args.path)
    
    # add a normalization layer
    model = add_normalization_layer(model, mean, std).to(device)

    # make sure the data is in [0,1] ! if you use pytorch ToTensor tranform it is already taken care of.
    # note we have already added a normalization layer to our models to adjust them to this data.
    dataset = CIFAR10_module(mean=(0,0,0), std=(1,1,1), data_dir = "./data", batch_size=args.batch_size)
    dataset.prepare_data()
    dataset.setup()

    #define attacks
    if args.attack == 'fgsm':
        Fgsm = torchattacks.FGSM(model, eps=args.eps)
        Fgsm = Torchattacks_Wrapper(Fgsm, 'Fgsm')
        attack = Fgsm
    elif args.attack == 'apgd-ce':
        Apgd_ce = AutoAttack(model, attacks_to_run = ['apgd-ce'], norm=args.norm, eps=args.eps, version='custom', verbose=False, device = device)
        Apgd_ce = AutoAttack_Wrapper(Apgd_ce, 'Apgd-ce')
        attack = Apgd_ce
    elif args.attack =='simple_apgd':
        simple_Apgd = torchattacks.APGD(model, norm='Linf', eps = 8/255, steps=10, loss='ce')
        simple_Apgd = Torchattacks_Wrapper(simple_Apgd, 'simple_APGD_10_iters')
        attack = simple_Apgd

    attacks = [
        attack
        ] 
    print()
    print('attacks used in training:')
    for atk in attacks: print(atk.get_name())
    print()

    # define  meaures
    loss_measure = Loss_measure(torch.nn.CrossEntropyLoss())
    normal_acc = Normal_accuracy()
    robust_acc = Robust_accuracy()
    feature_diff = Feature_diff(feature_layer_name)
    measurements = [normal_acc, robust_acc, loss_measure]#, feature_diff]

    #initialize and send the model to AdvLib
    model_bench = ab(model, device= device, tb_board_save_dir=tb_logs_path, predictor=lambda x: torch.max(x, 1)[1])

    # optimization params
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model[1].final_linear_layer.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    trainer = Robust_trainer(optimizer, loss)

    # train
    results = model_bench.train_val_test(trainer, args.max_epochs, dataset, measurements, attacks, model_save_path,
                                        train_measure_frequency=1, val_measure_frequency=1, run_test=False)

    print_train_test_val_result(results, measurements)
    save_training_results_to_csv(results, measurements, save_dir+'/results.csv')
    print('Verion is:', version)



    # Debug
    # print('encoder training: ', model[1].encoder.training)
    """ for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data) """



    

    


if __name__ == '__main__':
    parser = ArgumentParser()
    # model params
    parser.add_argument('model', type=str, choices=['barlow_twins', 'simCLR', 'supervised'], help='model type')
    parser.add_argument('device', type=int, help='cuda device number, e.g 2 means cuda:2') 
    parser.add_argument('path', type=str, help='path to encoder chekpoint') 
    parser.add_argument('--feature_num', type=int, help='number of output features for the unsupervised model, for resnet18 it is 512', default=512) 
    parser.add_argument('--class_num', type=int, help='number of classes' ,default=10)
    
    # data params
    parser.add_argument("--dataset", type=str, default="cifar10")

    # training params
    parser.add_argument('--batch_size', type=int, default=512, help='batch size for training the final linear layer')
    parser.add_argument("--optimizer", default="adam", type=str, choices=['adam'])        
    parser.add_argument("--max_epochs", default= 5, type=int, help="number of total epochs to run")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=0, type=float, help="weigth decay")

    #attack
    #parser.add_argument('--attack', action='append', help='give name of attack', required=True)
    parser.add_argument('--attack',type=str, help='name of attack', required=True, choices=['fgsm', 'apgd-ce', 'simple_apgd'])
    parser.add_argument('--eps',type=float, help='attack purturbation', default= 8/255)
    parser.add_argument('--norm',type=str, help='norm of attack', default='Linf', choices=['Linf'])


    args = parser.parse_args()

    """ args.model = 'supervised'
    args.path = './huy_Supervised_models_training_CIFAR10/cifar10/resnet18/version_3/checkpoints/best_val_acc_acc_val=88.37.ckpt'
    args.max_epochs = 20
    args.attack = 'simple_apgd'

    lrs = [1e-6, 1e-5, 1e-4]
    #grid search lr
    for lr in lrs:
        args.learning_rate = lr
        main(args) """


    main(args)

    # barlow_twins_model_path='./model_checkpoints/barlow_twins_unsupervised/0.0078125_128_128_cifar10_epoch_795.pth'
    # simCLR_model_path='./bolt_self_supervised_training/lightning_logs_simCLR_every5th_checkpoint/version_0/checkpoints/epoch=792-step=139567.ckpt'
    #supervised_path='./huy_Supervised_models_training_CIFAR10/cifar10/resnet18/version_3/checkpoints/best_val_acc_acc_val=88.37.ckpt'

    # python Top_linear_layer_adversarial_training.py supervised 0 './huy_Supervised_models_training_CIFAR10/cifar10/resnet18/version_3/checkpoints/best_val_acc_acc_val=88.37.ckpt' --attack 'fgsm'  --max_epochs 15 --learning_rate 1e-5
    # python Top_linear_layer_adversarial_training.py barlow_twins 0 './model_checkpoints/barlow_twins_unsupervised/0.0078125_128_128_cifar10_epoch_795.pth' --attack 'fgsm'  --max_epochs 5 --learning_rate 1e-5
    # python Top_linear_layer_adversarial_training.py simCLR 0 './bolt_self_supervised_training/lightning_logs_simCLR_every5th_checkpoint/version_0/checkpoints/epoch=792-step=139567.ckpt' --attack 'simple_apgd'  --max_epochs 20 --learning_rate 1e-5

    # good parameter: for FGSM and apgd and supervised backend it seems 1e-5 is the best learning rate, 
    # For FGSM barlow twins 1e-5 is too low, But for apgd .....
    # 
    #  
