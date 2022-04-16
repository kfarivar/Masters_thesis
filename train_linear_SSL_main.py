from argparse import ArgumentParser
from scipy.sparse import data
import torch
import torch.nn as nn
import torch.nn.functional as F


from lib.Get_dataset import CIFAR10_module, Causal_3Dident
#from models.barlow_twins_linear_classifier import BT_classifier
from models.SSL_linear_classifier import SSL_encoder_linear_classifier
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from models.models_mean_std import supervised_huy, barlow_twins_yao, simCLR_bolts

from copy import deepcopy



def main():
    parser = ArgumentParser()
    # model args
    parser = SSL_encoder_linear_classifier.add_model_specific_args(parser)
    args = parser.parse_args()

    # create the encoder + linear classification layer on top
    model = SSL_encoder_linear_classifier(**args.__dict__)

    # dataset specific modifications
    if args.model == 'barlow_twins':
        # normalization used in yao's barlow twins (the stds are off I should invetigate and retrain the model if necessary!)
        mean= simCLR_bolts[0]
        std= simCLR_bolts[1]
    
    elif args.model == 'simCLR':
        # taken from pl_bolts.transforms.dataset_normalizations.cifar10_normalization
        mean= simCLR_bolts[0]
        std= simCLR_bolts[1]

    elif args.model =='supervised':
        mean = supervised_huy[0]
        std = supervised_huy[1]

    elif args.model == 'simsiam':
        mean= simCLR_bolts[0]
        std= simCLR_bolts[1] 

        
    # prepare the dataset 
    # normalize the dataset same way as during unsupervised training
    if args.dataset == 'cifar10':
        dataset = CIFAR10_module(mean, std, batch_size=args.batch_size, augment_train=False)
        dataset.prepare_data()
        dataset.setup()
    elif args.dataset == '3dident':
        dataset = Causal_3Dident(data_dir='/home/kiarash_temp/adversarial-components/3dident_causal', 
                                augment_train=False, batch_size=args.batch_size, num_workers=16)
        dataset.setup()

    else:
        raise NotImplemented('the dataset you asked for is not supported.')



    # determin how to save the model
    if not args.regress_latents:
        checkpoint_callback = ModelCheckpoint(
                                monitor="val_acc",
                                filename="fixed_{model}_linear_layer_trained-{{epoch:02d}}-{{val_acc:.3f}}".format(model=args.model),
                                save_top_k=1,
                                mode="max"
                            )
    else:
        checkpoint_callback = ModelCheckpoint(
                                monitor="val_R^2/6", # spotlight is the 6th elment
                                filename="fixed_{model}_linear_layer_trained-{{epoch:02d}}-{{val_R^2/6:.3f}}".format(model=args.model),
                                save_top_k=1,
                                mode="max"
                            )


    # and train it on the train set.
    
    # Initialize a trainer
    trainer = Trainer(
        check_val_every_n_epoch= 1,
        gpus= [args.device],
        max_epochs= args.max_epochs,
        progress_bar_refresh_rate= 1, 
        default_root_dir= f'./{args.dataset}_last_layer_training_standard_logs/{args.model}_with_linear_layer_logs',
        callbacks= [checkpoint_callback],
        precision = 16,
        fast_dev_run=args.fast_dev_run
    )

    

    # Train the model 
    trainer.fit(model, dataset.train_dataloader(), dataset.test_dataloader())


    

# Do not train regression here ! it takes a long time and I couldn't get the r2 measure record all of features. 

if __name__ == '__main__':
    # barlow_twins_model_path='./model_checkpoints/barlow_twins_unsupervised/0.0078125_128_128_cifar10_epoch_795.pth'
    # simCLR_model_path='./bolt_self_supervised_training/lightning_logs_simCLR_every5th_checkpoint/version_0/checkpoints/epoch=792-step=139567.ckpt'
    #supervised_path='./huy_Supervised_models_training_CIFAR10/cifar10/resnet18/version_3/checkpoints/best_val_acc_acc_val=88.37.ckpt'

    # simsiam path = './bolt_self_supervised_training/simsiam/simsiam_resnet18_logs_and_chekpoints/lightning_logs/version_0/checkpoints/epoch=733-best_val_loss_val_loss=-0.9130538105964661.ckpt'
            

    # barlow twins command : python train_linear_SSL_main.py barlow_twins 1 ./model_checkpoints/barlow_twins_unsupervised/0.0078125_128_128_cifar10_epoch_795.pth

    # simCLR command: python train_linear_SSL_main.py simCLR 1 ./bolt_self_supervised_training/lightning_logs_simCLR_every5th_checkpoint/version_0/checkpoints/epoch=792-step=139567.ckpt
    
    # supervised command: python train_linear_SSL_main.py supervised 1 './huy_Supervised_models_training_CIFAR10/cifar10/resnet18/version_3/checkpoints/best_val_acc_acc_val=88.37.ckpt'
    
    # simsiam: python train_linear_SSL_main.py simsiam 3  './bolt_self_supervised_training/simsiam/simsiam_resnet18_logs_and_chekpoints/lightning_logs/version_0/checkpoints/epoch=733-best_val_loss_val_loss=-0.9130538105964661.ckpt'
    
    # (simclr all train used)
    #   python    train_linear_SSL_main.py simCLR 3  ./bolt_self_supervised_training/simclr/simCLR_resnet18_logs_and_chekpoints/lightning_logs/Using_all_of_train_set_400epochs/checkpoints/epoch=380_best_val_loss=5.988133430480957_online_val_acc=0.88.ckpt
    
    # supervised no augmentation
    #   python    train_linear_SSL_main.py supervised 5 /home/kiarash_temp/adversarial-components/huy_Supervised_models_training_CIFAR10/cifar10/resnet18/no_augmentations/checkpoints/best_val_acc_acc_val=82.27.ckpt
    
    # simclr supervised
    # python    train_linear_SSL_main.py simCLR 5 ./bolt_self_supervised_training/simclr/simCLR_resnet18_logs_and_chekpoints/lightning_logs/using_labels_version_4/epoch=462-best_val_loss_val_loss=3.899880886077881.ckpt

    # 3dident regression (doesn't work !)
    # python train_linear_SSL_main.py --dataset 3dident --regress_latents --batch_size 128 simCLR 2 ./bolt_self_supervised_training/simclr/simCLR_resnet18_logs_and_chekpoints/lightning_logs/version_4/checkpoints/epoch=411_best_val_loss=4.885046482086182_online_val_acc=1.00.ckpt
    
    # 3dident classify spotlight
    # python train_linear_SSL_main.py --dataset 3dident --classify_spotlight --batch_size 512 simCLR 6 ./bolt_self_supervised_training/simclr/simCLR_resnet18_logs_and_chekpoints/lightning_logs/version_4/checkpoints/epoch=411_best_val_loss=4.885046482086182_online_val_acc=1.00.ckpt
    
    
    # python train_linear_SSL_main.py --dataset 3dident --classify_spotlight --batch_size 512 --max_epochs 20 simCLR 6 ./bolt_self_supervised_training/simclr/3dident_simCLR_resnet18_logs_and_chekpoints/lightning_logs/small_arch/checkpoints/epoch=93_best_val_loss=6.664149761199951_online_val_acc=1.00.ckpt


    # python train_linear_SSL_main.py simCLR 4 /home/kiarash_temp/adversarial-components/bolt_self_supervised_training/simclr/cifar10_simCLR_resnet18_logs_and_chekpoints/lightning_logs/version_2/checkpoints/epoch=507_best_val_loss=5.973425388336182_online_val_acc=0.90.ckpt

    main()









""" for name, module in model.named_children():
            print()
            print(name)
            print(module) """

# save state dict of encoder to compare to after training (cheking it is fixed)
#before_train_encoder_dict = deepcopy(model.encoder.state_dict())

'''print('before encoder.training')
    print(model.encoder.training)'''

'''print('after encoder.training')
    print(model.encoder.training)
'''

""" # check if backend is fixed 
    print('after encoder.training')
    print(model.encoder.training)

    #after_dict  = model.encoders[0].cpu().state_dict()
    after_dict  = model.encoder.state_dict()
    diff = 0    
    for k,v in before_train_encoder_dict.items():
        #print(k)
        diff += (v-after_dict[k]).pow_(2).sum()
        #print((v-after_dict[k]).pow_(2).sum())

    print("diff is : " , diff) """