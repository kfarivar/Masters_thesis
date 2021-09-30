from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F


from lib.Get_dataset import CIFAR10_module
#from models.barlow_twins_linear_classifier import BT_classifier
from models.SSL_linear_classifier import SSL_encoder_linear_classifier
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

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
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]]
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]] 

        
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
        default_root_dir= f'./{args.model}_with_linear_layer_logs',
        callbacks= [checkpoint_callback],
        fast_dev_run=args.fast_dev_run
    )

    

    # Train the model 
    trainer.fit(model, dataset.train_dataloader(), dataset.test_dataloader())


    



if __name__ == '__main__':
    #barlow_twins model path: ./model_checkpoints/barlow_twins_unsupervised/0.0078125_128_128_cifar10_model_1000.pth
    # simCLR model path: ./model_checkpoints/simCLR_unsupervised/epoch=285-step=50335.ckpt

    #barlow twins command : python SSL_linear_main.py barlow_twins 1 ./model_checkpoints/barlow_twins_unsupervised/0.0078125_128_128_cifar10_model_1000.pth
        
    # simCLR command: python SSL_linear_main.py simCLR 1 ./model_checkpoints/simCLR_unsupervised/epoch=285-step=50335.ckpt
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