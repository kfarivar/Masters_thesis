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
from lib.utils import  print_train_test_val_result, Normalize_input, save_training_results_to_csv
from lib.Get_dataset import CIFAR10_module, Causal_3Dident
from lib.Measurements import Normal_accuracy, Robust_accuracy, Feature_diff, Loss_measure
from lib.Attacks import AutoAttack_Wrapper, Torchattacks_Wrapper, Identity_Wrapper
from lib.Trainer import Robust_trainer


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
    root_save_dir= f'./{args.dataset}_last_layer_training_robust_logs/{args.exp_name}_{args.model}_{args.attack}_linear_avdersarial_training_logs'
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
    if args.dataset == 'cifar10':
        if args.model == 'barlow_twins':
            # normalization used in yao's barlow twins (the stds are off I should invetigate and retrain the model if necessary!)
            mean= simCLR_bolts[0]
            std= simCLR_bolts[1]
            
        elif args.model == 'simCLR':
            # taken from pl_bolts.transforms.dataset_normalizations.cifar10_normalization
            mean= simCLR_bolts[0]
            std= simCLR_bolts[1]
            
        elif args.model == 'simsiam':
            mean= simCLR_bolts[0]
            std= simCLR_bolts[1]

        elif args.model =='supervised':
            mean = supervised_huy[0]
            std = supervised_huy[1]

        # make sure the data is in [0,1] ! if you use pytorch ToTensor tranform it is already taken care of.
        # note we have already added a normalization layer to our models to adjust them to this data.
        dataset = CIFAR10_module(mean=(0,0,0), std=(1,1,1), data_dir = "./data", batch_size=args.batch_size, num_workers=32)
        dataset.prepare_data()
        dataset.setup()

    elif args.dataset =='3dident':
        mean = [0.4327, 0.2689, 0.2839]
        std = [0.1201, 0.1457, 0.1082]

        dataset = Causal_3Dident(data_dir='/home/kiarash_temp/adversarial-components/3dident_causal', 
                                augment_train=True, no_normalization=True, batch_size=args.batch_size, num_workers=32, val_include_index=True,
                                train_subset=args.train_ratio, val_subset=args.val_ratio)
        dataset.setup()


    model = SSL_encoder_linear_classifier(args.model, args.path, dataset=args.dataset)
    # add a normalization layer
    model = Normalize_input(mean, std, model).to(device)

    

    #define attacks
    if args.attack == 'fgsm':
        attack_name = 'Fgsm'
        Fgsm = torchattacks.FGSM(model, eps=args.eps)
        Fgsm = Torchattacks_Wrapper(Fgsm, attack_name)
        attack = Fgsm

    elif args.attack == 'apgd-ce':
        attack_name =  'Apgd-ce'
        Apgd_ce = AutoAttack(model, attacks_to_run = ['apgd-ce'], norm=args.norm, eps=args.eps, version='custom', verbose=False, device = device)
        Apgd_ce = AutoAttack_Wrapper(Apgd_ce, attack_name)
        attack = Apgd_ce

    elif args.attack =='simple_apgd':
        attack_name = 'simple_APGD_10_iters'
        simple_Apgd = torchattacks.APGD(model, norm='Linf', eps = args.eps, steps=10, loss='ce')
        simple_Apgd = Torchattacks_Wrapper(simple_Apgd, attack_name)
        attack = simple_Apgd

    elif args.attack == 'fast_fgsm':
        attack_name = 'fast_Fgsm'
        fast_fgsm = torchattacks.FFGSM(model, eps=args.eps, alpha=10/255)
        fast_fgsm = Torchattacks_Wrapper(fast_fgsm, attack_name)
        attack = fast_fgsm
    
    elif args.attack == 'identity':
        attack_name = 'Identity_no_attack'
        attack = Identity_Wrapper()

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
    #feature_diff = Feature_diff(feature_layer_name)
    measurements = [normal_acc, robust_acc, loss_measure]#, feature_diff]

    #initialize and send the model to AdvLib
    model_bench = ab(model, device= device, tb_board_save_dir=tb_logs_path, predictor=lambda x: torch.max(x, 1)[1])

    # optimization params
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.model.final_linear_layer.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    trainer = Robust_trainer(optimizer, loss)

    # train
    results = model_bench.train_val_test(trainer, args.max_epochs, dataset, measurements, attacks, model_save_path, attack_name,
                                        train_measure_frequency=1, val_measure_frequency=1, run_test=False)

    #print_train_test_val_result(results, measurements)
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
    parser.add_argument('model', type=str, choices=['barlow_twins', 'simCLR', 'supervised', 'simsiam'], help='model type')
    parser.add_argument('device', type=int, help='cuda device number, e.g 2 means cuda:2') 
    # data params
    parser.add_argument("dataset", type=str)
    parser.add_argument('--train_ratio',type=float, default= 1)
    parser.add_argument('--val_ratio',type=float, default= 1)


    parser.add_argument('--path', type=str, required=True, help='path to encoder chekpoint') 
    parser.add_argument('--feature_num', type=int, help='number of output features for the unsupervised model, for resnet18 it is 512', default=512) 
    parser.add_argument('--class_num', type=int, help='number of classes' ,default=10)
    
    

    # training params
    parser.add_argument('--batch_size', type=int, default=512, help='batch size for training the final linear layer')
    parser.add_argument("--optimizer", default="adam", type=str, choices=['adam'])        
    parser.add_argument("--max_epochs", default= 20, type=int, help="number of total epochs to run")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=0, type=float, help="weigth decay")

    #attack
    #parser.add_argument('--attack', action='append', help='give name of attack', required=True)
    parser.add_argument('--attack', default='simple_apgd', type=str, help='name of attack', required=True, choices=['fgsm', 'fast_fgsm', 'apgd-ce', 'simple_apgd', 'identity'])
    parser.add_argument('--eps',type=float, help='attack purturbation', default= 8/255)
    parser.add_argument('--norm',type=str, help='norm of attack', default='Linf', choices=['Linf'])

    parser.add_argument('--exp_name', type=str, default='Experiment_name', help='experiment name')
    

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


    # python Top_linear_layer_adversarial_training.py supervised 0 './huy_Supervised_models_training_CIFAR10/cifar10/resnet18/version_3/checkpoints/best_val_acc_acc_val=88.37.ckpt' --attack 'fgsm'  --max_epochs 15 --learning_rate 1e-5
    # python Top_linear_layer_adversarial_training.py barlow_twins 0 './model_checkpoints/barlow_twins_unsupervised/0.0078125_128_128_cifar10_epoch_795.pth' --attack 'fgsm'  --max_epochs 5 --learning_rate 1e-5
    # python Top_linear_layer_adversarial_training.py simCLR 0 './bolt_self_supervised_training/lightning_logs_simCLR_every5th_checkpoint/version_0/checkpoints/epoch=792-step=139567.ckpt' --attack 'simple_apgd'  --max_epochs 20 --learning_rate 1e-5
    # python Top_linear_layer_adversarial_training.py simsiam 3 './bolt_self_supervised_training/simsiam/simsiam_resnet18_logs_and_chekpoints/lightning_logs/version_0/checkpoints/epoch=733-best_val_loss_val_loss=-0.9130538105964661.ckpt'  --attack 'simple_apgd'  --max_epochs 20 --learning_rate 1e-4

    #simCLR all train
    # python Top_linear_layer_adversarial_training.py simCLR 4  ./bolt_self_supervised_training/simclr/simCLR_resnet18_logs_and_chekpoints/lightning_logs/Using_all_of_train_set_400epochs/checkpoints/epoch=380_best_val_loss=5.988133430480957_online_val_acc=0.88.ckpt --attack 'simple_apgd'  --max_epochs 20 --learning_rate 1e-5

    # supervised simclr
    # python Top_linear_layer_adversarial_training.py simCLR 4 --exp_name 'supervised' --attack 'simple_apgd'  --max_epochs 20 --learning_rate 1e-5 --path ./bolt_self_supervised_training/simclr/simCLR_resnet18_logs_and_chekpoints/lightning_logs/using_labels_version_4/epoch=462-best_val_loss_val_loss=3.899880886077881.ckpt  


    # good parameter: for FGSM and apgd and supervised backend it seems 1e-5 is the best learning rate, 
    # For FGSM barlow twins 1e-5 is too low, for apgd 1e-4
    # for SSL models try 1e-3, 1e-4, 1e-5, 1e-6
    #  

    #python Top_linear_layer_adversarial_training.py simCLR 4 3dident --train_ratio 0.04 --val_ratio 0.04 --attack 'simple_apgd' --exp_name 'first_try_' --learning_rate 1e-4 --path ./bolt_self_supervised_training/simclr/3dident_simCLR_resnet18_logs_and_chekpoints/lightning_logs/small_arch/checkpoints/epoch=93_best_val_loss=6.664149761199951_online_val_acc=1.00.ckpt
