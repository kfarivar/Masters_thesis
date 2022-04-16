from argparse import ArgumentParser
import glob
from argparse import ArgumentParser
import json

import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from torch.utils import data
from torchmetrics import Accuracy, R2Score
#from sklearn.metrics import r2_score

from barlow_twins_yao_training.barlowtwins_module import BarlowTwins
from huy_Supervised_models_training_CIFAR10.module import CIFAR10Module
from bolt_self_supervised_training.simclr.simclr_module import SimCLR
from bolt_self_supervised_training.simsiam.simsiam_module import SimSiam

from huy_Supervised_models_training_CIFAR10.module import Causal3DidentModel


class SSL_encoder_linear_classifier(LightningModule):
    ''' This is an self-supervised learning module + linear classifier added to the last layer.
        So we can use the SSL encoder for classification. 
        It also includes the info about training and optimizer. 
        This can also be a regressor in case of 3dident.
    '''
    def __init__(self, model, path, dataset='cifar10', regress_latents=False, classify_spotlight=False, feature_num=512, class_num=10, optimizer=None, learning_rate=1, loading=False, **kwargs):
        ''' model: the type of SSL model to use as encoder
            path: the chekpoint for the best model chekpoint
            feature_num: number of output features for the unsupervised model
            class_num: number of classes, if regress_latents then the number of latents to regress.
            regress_latents: in case of 3dident whether to classify objects or regress latents.
        '''
        super().__init__()
        if not loading:
            self.save_hyperparameters()
        self.optim = optimizer
        self.lr = learning_rate

        self.dataset = dataset
        self.regress_latents = regress_latents
        self.classify_spotlight = classify_spotlight

        #the masurement is accuracy
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        
        """ # or coefficient of determination (hacky), seperated to be able to report them individually.
        self.train_r2_scores = []
        self.val_r2_scores = []
        for i in range(class_num):
            self.train_r2_scores.append(R2Score())
            self.val_r2_scores.append(R2Score())
        # register the metrics properly as a child.
        self.train_r2_scores = nn.ModuleList(self.train_r2_scores)
        self.val_r2_scores = nn.ModuleList(self.val_r2_scores) """

        """ self.train_spotlight_R2 = R2Score()
        self.val_spotlight_R2 = R2Score() """


        if dataset == '3dident' and classify_spotlight:
            class_num = 3
        else:
            class_num = 10
        

        # disable the gradient of encoder and put it in eval mode
        self.encoder = Encoder(model, path, dataset)
        # add a linear layer (#features to #classes) 
        self.final_linear_layer = nn.Linear(feature_num, class_num)

    def forward(self, x):
        features = self.encoder(x)
        return self.final_linear_layer(features)

    def shared_step(self, batch, mode):
        # note for each batch we get: 
        # cifar: data, target, index 
        # 3dident: data, target, latents
        if self.dataset == 'cifar10' :
            x, y, _ = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)

            self.log(f'{mode}_classification_loss', loss, on_step=False, on_epoch=True)

            if mode == 'train':
                acc = self.train_acc(y_hat, y)
            else:
                acc = self.val_acc(y_hat, y)
            self.log(f'{mode}_acc', acc, on_step=False, on_epoch=True)
        
        elif self.classify_spotlight:
            images, _, latents = batch

            # get classes
            labels = Causal3DidentModel.spotlight_label_from_latent(latents)
            labels = labels.to(images.device)
            # filter outputs 
            images = images[labels != -1]
            # filter labels
            labels = labels[labels != -1]

            outputs = self(images)

            _, predictions = torch.max(outputs, 1)
            loss = F.cross_entropy(outputs, labels)
            self.log(f'{mode}_classification_loss', loss) #, on_step=True, on_epoch=True)

            if mode == 'train':
                acc = self.train_acc(predictions, labels)
            else:
                acc = self.val_acc(predictions, labels)
            self.log(f'{mode}_acc', acc) #, on_step=True, on_epoch=True)


        """ There is a problem with speed and logging gives error
        elif self.dataset == '3dident' and self.regress_latents:
            x, object_class, latents = batch
            y_hat = self(x)
            loss = F.mse_loss(y_hat, latents)

            self.log(f'{mode}_regession_loss', loss, on_step=False, on_epoch=True)

            if mode == 'train':
                for i, r2_score in enumerate(self.train_r2_scores):
                    r2_score(y_hat[:,i], latents[:,i])
                    self.log(f'train_R^2/{i}', r2_score, on_step=False, on_epoch=True)
            else: 
                for i, r2_score in enumerate(self.val_r2_scores):
                    r2_score(y_hat[:,i], latents[:,i])
                    self.log(f'val_R^2/{i}', r2_score, on_step=False, on_epoch=True)
            
            if mode =='train':
                r2 = self.train_spotlight_R2(y_hat[:,6], latents[:,6])
            else:
                r2 = self.val_spotlight_R2(y_hat[:,6], latents[:,6])
            
            self.log(f'{mode}_R^2_spotlight', r2, on_step=False, on_epoch=True) 

        else:
            raise NotImplementedError('dataset or regression type not implemented !') """


        return loss


    def training_step(self, batch, batch_nb):

        return self.shared_step(batch, mode='train')
        
    def validation_step(self, batch, batch_idx):

        return self.shared_step(batch, mode='val')


    def configure_optimizers(self):
        # only optimize the linear layer !
        if self.optim == 'adam':
            return torch.optim.Adam(self.final_linear_layer.parameters(), lr=self.lr)
        else:
            raise NotImplemented('that optimizer is not implemented.')

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        mode,
        from_version=True,
        map_location = None,
        hparams_file = None,
        strict: bool = True,
        **kwargs,
    ):
        '''I overwrite the load for easier loading. 

            mode: 'standard' or 'robust' determins how to choose the checkpoint

            from_version: if True It is enough to specify the version. I will choose the best checkpoint. 
                          if False you can pass a specific checkpoint from a version.
        '''

        if from_version:
            if mode == 'standard':
                # go to chekpoint choose the only checpoint and load it.
                # lightining uses the hparam file to initialize the model which loades the encoder automatically.
                chkpts = glob.glob(checkpoint_path+"/checkpoints/*")
                if len(chkpts)==1:
                    print('Loading: ' + chkpts[0])
                    return super().load_from_checkpoint(chkpts[0], map_location, hparams_file, strict, **kwargs)
                elif len(chkpts)==0:
                    raise FileNotFoundError('No checkpoints were found !')
                else:
                    raise NotImplementedError('Multiple checkpoints found. This standard loading is not implemented')
            
            elif mode == 'robust':
                # this correspnds to my libraries results
                # look for the checkpoint starting with best in its name
                chkpts = glob.glob(checkpoint_path+"/checkpoints/best*.pt")
                if len(chkpts)==1:
                    chpt = chkpts[0]
                    print("loading: " + chpt)
                    # find the hparams file and load the encoder network
                    hparams_path = checkpoint_path + '/commandline_args.txt'
                    with open(hparams_path, 'r') as f:
                        hparams = json.load(f)
                    
                    # load the encoder (for some reason running save_hparams throws an error, thats why I added the loading flag.)
                    model = cls(hparams['model'], hparams['path'], loading=True)
                    # load the linear layer
                    model_dict = torch.load(chpt) 
                    bias = model_dict['model.final_linear_layer.bias']
                    weights = model_dict['model.final_linear_layer.weight']
                    model.final_linear_layer.weight = torch.nn.Parameter(weights)
                    model.final_linear_layer.bias = torch.nn.Parameter(bias)

                    return model

                elif len(chkpts)==0:
                    raise FileNotFoundError('No checkpoints were found !')
                    
                else:
                    raise NotImplementedError('This robust loading is not implemented')


        else:
            if mode == 'standard':
                return super().load_from_checkpoint(checkpoint_path, map_location, hparams_file, strict, **kwargs)
            
            elif mode == 'robust':
                raise NotADirectoryError()

        


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # model params
        parser.add_argument('model', type=str, choices=['barlow_twins', 'simCLR', 'simsiam', 'supervised'], help='model type')
        parser.add_argument('device', type=int, help='cuda device number, e.g 2 means cuda:2') 
        parser.add_argument('path', type=str, help='path to model chekpoint') 
        parser.add_argument('--feature_num', type=int, help='number of output features for the unsupervised model, for resnet18 it is 512', default=512) 
        parser.add_argument('--class_num', type=int, help='number of classes' ,default=10)

        parser.add_argument('--regress_latents', action='store_true', help='used for regressing 3dident')
        parser.add_argument('--classify_spotlight', action='store_true', help='used for classifying 3dident apotlight')
        
        # transform params
        parser.add_argument("--dataset", type=str, default="cifar10")

        # training params
        parser.add_argument('--batch_size', type=int, default=512, help='batch size for training the final linear layer')
        parser.add_argument("--optimizer", default="adam", type=str, choices=['adam'])        
        parser.add_argument("--max_epochs", default= 5, type=int, help="number of total epochs to run")
        parser.add_argument("--learning_rate", default=1e-2, type=float, help="learning rate")

        #fast_dev_run
        #This flag runs a “unit test” by running n if set to n (int) else 1 if set to True 
        # training and validation batch(es). The point is to detect any bugs in the training/validation loop 
        # without having to wait for a full epoch to crash.
        parser.add_argument("--fast_dev_run", action='store_true', default=False)
        


        return parser




from pl_bolts.models.self_supervised.resnets import resnet18 as lit_ssl_resnet18

class huy_supervised(LightningModule):
    '''only load the encoder part of the supervised model to train the last layer like other SSL models'''
    def __init__(self, dataset, classifier='resnet18'):
        #super().__init__(classifier=classifier)
        super().__init__()
        if dataset == 'cifar10':
            self.model = lit_ssl_resnet18(first_conv=False, maxpool1=False, return_all_feature_maps=False)
        elif dataset == '3dident':
            self.model = lit_ssl_resnet18(first_conv=True, maxpool1=True, return_all_feature_maps=False)

    def forward(self, x):
        # exclude the last linear layer
        # bolt models return a list
        return self.model(x)[-1]

    
from collections import OrderedDict

class Encoder(LightningModule):
    '''This is the model (mostly Resnet18) that has been trained using unsupervised learning.
        The point of the module is to load the encoder part of the main model and keep the model's
        requires_grad to False and in eval mode. 
    '''

    def __init__(self, model, path, dataset):
        super().__init__()
        self.model = model
        # load pretrained unsupervised model
        # Important: in lightining the 'load_from_checkpoint' method ,unlike pytorch, returns the loaded model 
        # IN LIGHTINING LOADING DOESN'T HAPPEN IN PLACE, IT IS RETURNED !! 

        # For lightining models
        # we need to use lightinings own loading method, there is a top linear layer added durin unsupervised learning 
        # and setting strict to False ignores that.("non_linear_evaluator.block_forward.2.weight", "non_linear_evaluator.block_forward.2.bias".)
        # the forward method only applies the encoder and not the projector. so no need to call encoder.

        # Possible improvement For some of the models the forward calculates the unused one layer projection(s) as well which might slightly slow them down. 

        if model == 'barlow_twins':
            # lightining
            encoder = BarlowTwins.load_from_checkpoint(path, strict=False)
            self.encoder = encoder
            # flatten output of encoder 
            self.pre_process = nn.Flatten()
            

        elif model == 'simCLR':
            # lightining
            encoder = SimCLR.load_from_checkpoint(path, strict=False)
            self.encoder = encoder
            self.pre_process = nn.Flatten()

        elif model =='simsiam':
            #lightining 
            encoder = SimSiam.load_from_checkpoint(path, strict=False)
            self.encoder = encoder
            self.pre_process = nn.Flatten()

        elif model == 'supervised':
            encoder = huy_supervised.load_from_checkpoint(path, dataset=dataset, strict=False)
            self.encoder = encoder
            self.pre_process = nn.Flatten()

        else:
            raise NotImplementedError('This encoder for SSL is not supported yet.')
        
        
        self.freeze()


    def forward(self,x):
        encoder_result = self.encoder(x)
        features = self.pre_process(encoder_result)

        return features  

    def setup(self, device: torch.device):
        '''makes all requires_grad of parameters False and sets the model in eval mode.'''
        self.freeze()

    def train(self, mode: bool):
        ''' avoid pytorch lighting auto set trian mode. keep it in eval. '''
        return super().train(False)

    def requires_grad_(self, requires_grad: bool):
        return super().requires_grad_(False)

    def state_dict(self, destination, prefix, keep_vars):
        ''' (probably needs to be fixed!) avoid pytorch lighting auto save params '''
        destination = OrderedDict()
        destination._metadata = OrderedDict()
        return destination