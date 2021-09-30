''' This is barlow twins encoder + linear classifier added to the last layer 
    and the info about training and optimizer. 
'''
import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule

from .barlow_twins_with_projector import BT
from lib.utils import normalize, remove_operation_count_from_dict

class BT_classifier(LightningModule):
    def __init__(self, path, feature_num, class_num):
        ''' path: the chekpoint for the best barlow teins chekpoint
            feature_num: number of output features for the unsupervised model
            class_num: number of classes
        '''
        super().__init__()
        self.save_hyperparameters()

        # flatten output of encoder and normalize
        pre_process = nn.Sequential(nn.Flatten(), normalize(dim=-1))
        # disable the gradient of encoder and put it in eval mode
        self.encoder = Encoder(path, pre_process)
        # add a linear layer (#features to #classes) 
        self.final_linear_layer = nn.Linear(feature_num, class_num)



    def forward(self, x):
        features = self.encoder(x)
        return self.final_linear_layer(features)

    def training_step(self, batch, batch_nb):
        # note we get: data, target, index for each batch.
        x, y, _ = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        # only optimize the linear layer !
        return torch.optim.Adam(self.final_linear_layer.parameters(), lr=0.01)


    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        pred_result = (torch.max(y_hat, dim=1)[1] == y)
        return pred_result, loss


    def validation_epoch_end(self, validation_step_outputs):
        #print(validation_step_outputs)
        num_points = 0
        epoch_loss_sum = 0
        epoch_correct_sum = 0
        for pred_result, loss in validation_step_outputs:
            num_points += pred_result.shape[0]
            epoch_loss_sum += torch.sum(loss)
            epoch_correct_sum += torch.sum(pred_result)

        self.log('val_loss', epoch_loss_sum/num_points)
        self.log('val_accuracy', epoch_correct_sum/num_points)

            

 



from collections import OrderedDict

class Encoder(LightningModule):
    '''This is the Resnet that has been trained using unsupervised learning.
    '''

    def __init__(self, path, pre_process):
        super().__init__()
        # load pretrained unsupervised model
        bt = BT()
        # if key ends in total_ops or total_params remove it.
        state_dict = remove_operation_count_from_dict(torch.load(path))
        bt.load_state_dict(state_dict)

        self.model = nn.Sequential(
            bt.f, # only use the encoder
            pre_process
        )
        self.freeze()

    def forward(self,x):
        return self.model(x) 

    def setup(self, device: torch.device):
        # makes all requires_grad of parameters False and sets the model in eval mode.
        self.freeze()

    def train(self, mode: bool):
        """ avoid pytorch lighting auto set trian mode """
        return super().train(False)

    def state_dict(self, destination, prefix, keep_vars):
        ''' (probably needs to be fixed!) avoid pytorch lighting auto save params '''
        destination = OrderedDict()
        destination._metadata = OrderedDict()
        return destination