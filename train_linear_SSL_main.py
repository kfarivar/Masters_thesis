from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F


from lib.Get_dataset import CIFAR10_module
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
        mean=(0.4914, 0.4822, 0.4465)
        std=(0.2023, 0.1994, 0.2010)
    
    if args.model == 'simCLR':
        # taken from pl_bolts.transforms.dataset_normalizations.cifar10_normalization
        mean= simCLR_bolts[0]
        std= simCLR_bolts[1]

    if args.model =='supervised':
        mean = supervised_huy[0]
        std = supervised_huy[1]

        
    # prepare the dataset 
    # normalize the dataset same way as during unsupervised training
    if args.dataset == 'cifar10':
        dataset = CIFAR10_module(mean, std, batch_size=args.batch_size, augment_train=False)
        dataset.prepare_data()
        dataset.setup()
    else:
        raise NotImplemented('the dataset you asked for is not supported.')



    # determin how to save the model
    checkpoint_callback = ModelCheckpoint(
                            monitor="val_acc",
                            filename="fixed_{model}_linear_layer_trained-{{epoch:02d}}-{{val_acc:.3f}}".format(model=args.model),
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
        default_root_dir= f'./last_layer_training_standard_logs/{args.model}_with_linear_layer_logs',
        callbacks= [checkpoint_callback],
        fast_dev_run=args.fast_dev_run
    )

    

    # Train the model 
    trainer.fit(model, dataset.train_dataloader(), dataset.test_dataloader())


    



if __name__ == '__main__':
    # barlow_twins_model_path='./model_checkpoints/barlow_twins_unsupervised/0.0078125_128_128_cifar10_epoch_795.pth'
    # simCLR_model_path='./bolt_self_supervised_training/lightning_logs_simCLR_every5th_checkpoint/version_0/checkpoints/epoch=792-step=139567.ckpt'

    #supervised_path='./huy_Supervised_models_training_CIFAR10/cifar10/resnet18/version_3/checkpoints/best_val_acc_acc_val=88.37.ckpt'

    #barlow twins command : python train_linear_SSL_main.py barlow_twins 1 ./model_checkpoints/barlow_twins_unsupervised/0.0078125_128_128_cifar10_epoch_795.pth

    # simCLR command: python train_linear_SSL_main.py simCLR 1 ./bolt_self_supervised_training/lightning_logs_simCLR_every5th_checkpoint/version_0/checkpoints/epoch=792-step=139567.ckpt
    
    #supervised command: python train_linear_SSL_main.py supervised 1 './huy_Supervised_models_training_CIFAR10/cifar10/resnet18/version_3/checkpoints/best_val_acc_acc_val=88.37.ckpt'
    
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