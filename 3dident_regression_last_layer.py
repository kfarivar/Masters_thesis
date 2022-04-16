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


def train(args):

    # train the readout on subset of val and evaluate it on  another subset

    # get encoder
    encoder = Encoder(args.model, args.path, '3dident').to(args.device)

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
    train_classes =  torch.cat(train_classes, 0)

    val_latents = torch.cat(val_latents, 0)
    val_features = torch.cat(val_features, 0)
    val_classes = torch.cat(val_classes, 0)

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
        path = './bolt_self_supervised_training/simclr/3dident_simCLR_resnet18_logs_and_chekpoints/lightning_logs/version_3/checkpoints/epoch=93_best_val_loss=6.664149761199951_online_val_acc=1.00.ckpt'
        batch_size = 512
        num_workers = 32
        num_train_batches = 5
        num_eval_batches = 10
        device = 'cuda:4'




    args = Args()
    train(args)

    # The non-linear result is worse since I sucpect the hiperparameter search range is not wide enough ! (shouldn't matter just use linear !)
    '''
    render internally assumes the variables form these value ranges:
        
        per object:
            0. x position in [-3, -3]
            1. y position in [-3, -3]
            2. z position in [-3, -3]
            3. alpha rotation in [0, 2pi]
            4. beta rotation in [0, 2pi]
            5. gamma rotation in [0, 2pi]
            6. theta spot light in [0, 2pi]
            7. hue object in [0, 2pi]
            8. hue spot light in [0, 2pi]
        
        per scene:
            9. hue background in [0, 2pi]
    '''

    '''Results: (the first value is the classification result)

    Model without first conv reducing resolution (224x224)
    results linear:
    [0.999609375, 0.8182625545484512, 0.7527109820166971, 0.8107530437408981, 0.7071569836748409, 0.662349503227164, 0.6094832399531038, 0.9398602327770723, -0.18928967877907432, -0.21775831929535183, -0.008193754552185739]

    non linear:
    [0.999609375, 0.53140714932498, 0.4527231056397335, 0.5646079908095339, 0.5629575334637231, 0.5629968308922161, 0.5194286039945086, 0.7712245813636629, -0.08238430369045635, -0.001444778770504307, -0.00037729550170384485]


    Model with reducing first layer (112x112)
    results linear:
    [1.0, 0.8639570873788722, 0.7912421253663532, 0.7916736909965967, 0.6205633304872211, 0.5270109695267875, 0.5334545237997002, 0.9620300509820501, -0.23423116776860775, -0.2099583065112378, -0.14693430834798993]

    non linear:
    [1.0, 0.7295743842761302, 0.6579604724328898, 0.670759680802943, 0.5815572339950723, 0.49690154398165587, 0.5688862883373637, 0.9216010758158096, -0.08694263492322163, 2.3596646643087027e-05, -0.0010522516267132964]



    '''



