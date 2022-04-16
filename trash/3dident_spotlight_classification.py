from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model, kernel_ridge
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

from lib.Get_dataset import Causal_3Dident

from models.SSL_linear_classifier import Encoder

import sys
sys.path.append('/home/kiarash_temp/adversarial-components/3dident_causal/ssl_identifiability/')
import disentanglement_utils


from huy_Supervised_models_training_CIFAR10.module import Causal3DidentModel



def train(args):

    # train the readout on subset of val and evaluate it on  another subset

    # get encoder
    encoder = Encoder(args.model, args.path).to(args.device)

    # get the features from images and save them in array
    dataset = Causal_3Dident(data_dir='/home/kiarash_temp/adversarial-components/3dident_causal', 
                                augment_train=False, batch_size=args.batch_size, num_workers=args.num_workers)
    dataset.setup()

    train_features = []
    train_latents = []
    train_classes = []

    val_features = []
    val_latents = []
    val_classes = []

    print('encoding ....')
    for i, (x, object_class, latents) in enumerate(dataset.train_dataloader()):
        x, object_class, latents = x.to(args.device), object_class.to(args.device), latents.to(args.device)
        y_hat = encoder(x)
        if i < args.num_train_batches:
            train_features.append(y_hat)
            train_latents.append(latents)

            train_classes.append(object_class)
        else:
            break

    for i, (x, object_class, latents) in enumerate(dataset.val_dataloader()):
        x, object_class, latents = x.to(args.device), object_class.to(args.device), latents.to(args.device)
        y_hat = encoder(x)
        if i < args.num_eval_batches:
            val_features.append(y_hat)
            val_latents.append(latents)

            val_classes.append(object_class)
        else:
            break
    

    train_features =  torch.cat(train_features, 0)
    train_latents =  torch.cat(train_latents, 0)
    

    val_latents = torch.cat(val_latents, 0)
    val_features = torch.cat(val_features, 0)
    

    print('training ...')
    # SCALE features
    scaler_hz = StandardScaler()
    hz = scaler_hz.fit_transform(train_features.detach().cpu().numpy())
    # train for latents
    n_models = []
    l_models = []
    scaler_zs = []

    for i in range(train_latents.size(-1)):
        scaler_z = StandardScaler()
        standardized_z = scaler_z.fit_transform(torch.reshape(train_latents[:,i],
                                                        (-1,1)).detach().cpu().numpy())

        scaler_zs.append(scaler_z)

        n_models.append(disentanglement_utils.nonlinear_disentanglement(
            standardized_z, hz, train_mode=True
        ))
        l_models.append(disentanglement_utils.linear_disentanglement(
            standardized_z, hz, train_mode=True
        ))

    # object classification
    log_model = disentanglement_utils.linear_disentanglement(train_classes, hz, 
                                                                train_mode=True, 
                                                                mode="accuracy")

    print('evaluating ... ')
    # evaluation
    hz = scaler_hz.transform(val_features.detach().cpu().numpy())
    lin_scores = []
    nonlin_scores = []
    for i in range(val_latents.size(-1)):
        scaled_zi = scaler_zs[i].transform(torch.reshape(val_latents[:,i], 
                                                    (-1,1)).detach().cpu().numpy())

        (linear_disentanglement_score,_,), _ = disentanglement_utils.linear_disentanglement(
            scaled_zi, hz, 
            mode="r2", model=l_models[i]
        )
        lin_scores.append(linear_disentanglement_score)


        (
            nonlinear_disentanglement_score,
            _,
        ), _ = disentanglement_utils.nonlinear_disentanglement(
            scaled_zi, hz, 
            mode="r2", model=n_models[i]
        )
        nonlin_scores.append(nonlinear_disentanglement_score)


    (
        log_score,
        _,
    ), _ = disentanglement_utils.linear_disentanglement(val_classes, 
                                                                        hz, 
                                                                mode="accuracy", 
                                                            model=log_model)
    lin_scores.insert(0, log_score)
    nonlin_scores.insert(0, lin_scores[0])


    print()
    print('results linear:')
    print(lin_scores)
    
    print()
    print('non linear:')
    print(nonlin_scores)





    # use scikit learn to fit a model


if __name__ == '__main__':
    from dataclasses import dataclass
    @dataclass
    class Args:
        model = 'simCLR'
        path = './bolt_self_supervised_training/simclr/simCLR_resnet18_logs_and_chekpoints/lightning_logs/version_4/checkpoints/epoch=411_best_val_loss=4.885046482086182_online_val_acc=1.00.ckpt'
        batch_size = 512
        num_workers = 32
        num_train_batches = 5
        num_eval_batches = 10
        device = 'cuda:2'


    args = Args()
    train(args)