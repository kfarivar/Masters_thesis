from argparse import ArgumentParser
import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy

from barlow_twins_yao_training.model import Model as BT
from huy_Supervised_models_training_CIFAR10.module import CIFAR10Module
from .simclr_module import SimCLR
from lib.utils import normalize, remove_operation_count_from_dict

class SSL_encoder_linear_classifier(LightningModule):
    ''' This is an self-supervised learning module + linear classifier added to the last layer.
        So we can use the SSL encoder for classification. 
        It also includes the info about training and optimizer. 
    '''
    def __init__(self, model, path, feature_num=512, class_num=10, optimizer=None, learning_rate=1, **kwargs):
        ''' model: the type of SSL model to use as encoder
            path: the chekpoint for the best model chekpoint
            feature_num: number of output features for the unsupervised model
            class_num: number of classes
        '''
        super().__init__()
        self.save_hyperparameters()
        self.optim = optimizer
        self.lr = learning_rate

        #the masurement is accuracy
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

        # disable the gradient of encoder and put it in eval mode
        self.encoder = Encoder(model, path)
        # add a linear layer (#features to #classes) 
        self.final_linear_layer = nn.Linear(feature_num, class_num)

    def forward(self, x):
        features = self.encoder(x)
        return self.final_linear_layer(features)

    def training_step(self, batch, batch_nb):
        # note we get: data, target, index for each batch.
        x, y, _ = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.train_acc(y_hat, y)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)

        return loss


    def configure_optimizers(self):
        # only optimize the linear layer !
        if self.optim == 'adam':
            return torch.optim.Adam(self.final_linear_layer.parameters(), lr=self.lr)
        else:
            raise NotImplemented('that optimizer is not implemented.')


    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.val_acc(y_hat, y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)

        


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # model params
        parser.add_argument('model', type=str, choices=['barlow_twins', 'simCLR', 'BYOL', 'supervised'], help='model type')
        parser.add_argument('device', type=int, help='cuda device number, e.g 2 means cuda:2') 
        parser.add_argument('path', type=str, help='path to model chekpoint') 
        parser.add_argument('--feature_num', type=int, help='number of output features for the unsupervised model, for resnet18 it is 512', default=512) 
        parser.add_argument('--class_num', type=int, help='number of classes' ,default=10)
        
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




class barlow_twins(BT):
    '''correct the forward method of the barlow twins model'''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # apply encoder
        # unpack the bolt resnet result
        return self.f(x)[-1]
 
class huy_supervised(CIFAR10Module):
    '''extract the encoder part of the supervised model to train the last layer like other SSL models'''
    def __init__(self, classifier='resnet18'):
        super().__init__(classifier=classifier)

    def forward(self, x):
        # exclude the last linear layer
        return self.model(x)[-1]


from collections import OrderedDict

class Encoder(LightningModule):
    '''This is the model (mostly Resnet18) that has been trained using unsupervised learning.
        The point of the module is to load the encoder part of the main model and keep the model's
        requires_grad to False and in eval mode. 
    '''

    def __init__(self, model, path):
        super().__init__()
        self.model = model
        # load pretrained unsupervised model
        if model == 'barlow_twins':
            encoder = barlow_twins()
            # if key ends in total_ops or total_params remove it.(only needed for barlow twins)
            state_dict = remove_operation_count_from_dict(torch.load(path))
            # in pytorch we can load in place.
            encoder.load_state_dict(state_dict)
            self.encoder = encoder
            # flatten output of encoder and normalize (since yao's implementation normalizes during training)
            self.pre_process = nn.Sequential(nn.Flatten())#, normalize(dim=-1))

        elif model == 'simCLR':
            # Important: in lightining the 'load_from_checkpoint' method ,unlike pytorch, returns the loaded model 
            # IN LIGHTINING LOADING DOESN'T HAPPEN IN PLACE, IT IS RETURNED !! 
            # we need to use lightinings own loading method, there is a top linear layer added durin unsupervised learning 
            # and setting strict to False ignores that.("non_linear_evaluator.block_forward.2.weight", "non_linear_evaluator.block_forward.2.bias".)
            encoder = SimCLR.load_from_checkpoint(path, strict=False)
            # the forward method only applies the encoder and not the projector. so no need to call encoder.
            self.encoder = encoder
            self.pre_process = nn.Flatten()

        elif model == 'supervised':
            encoder = huy_supervised.load_from_checkpoint(path, strict=False)
            self.encoder = encoder
            self.pre_process = nn.Flatten()

        else:
            raise NotImplemented('This encoder for SSL is not supported yet.')
        
        
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