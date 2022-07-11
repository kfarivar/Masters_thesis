import torch
from torch import nn
from torch._C import dtype
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, dataset, random_split

import random
import numpy as np


import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

import torchattacks




class optimal_projector(pl.LightningModule):
    def __init__(self, epsilon):
        super().__init__()

        # hand tune the conv layer
        with torch.no_grad():
            # The input is assumed to be a 1 channel 0/1 image of size 10x10. only containing one of the patterns which can be shifted inside the image frame !
            self.conv1 = nn.Conv2d(1, 2, 5)

            # the amount of adv perturbation
            # This value should not be 0.5 or more otherwise with 0/1 images 
            # the adversary will be able to create or remove photos 
            # in the sense that given a shape with all 0.5 for pixels 
            # we don't know if this is created by adv or faded by adv !
            if 0<epsilon and epsilon < 0.5:
                self.epsilon = epsilon
            else:
                raise ValueError('epsilon is out of range !')

            ###################
            # The triangle

            # number of pixels used in triangle shape and its conv filter
            # used for checking the bounds of activation and preventing adversary to fool us
            triangle_num_pixels = 16
            # I normalize the filter by number of 1 pixels in the pattern
            # so if all pixels are active I get 1. This makes the threshold for epsilon 0.5 and makes proving robustness easier.

            # I need to check two inequalities with 2 different convolutional transformations (the sign and the bias are diff)
            # So for each shape I need two filters

            self.conv1.weight[0] =  torch.nn.Parameter(
                    (1/triangle_num_pixels) * torch.tensor([
                        [0, 1, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1]
                    ])
                )

            
            
            #################
            # square 
            square_num_pixels = 16

            self.conv1.weight[1] =  torch.nn.Parameter(
                (1/square_num_pixels) * torch.tensor([
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]
            ]) )

            # define the boundries for chekcing the adv perturbs
            # inbound becomes 0 out of bound some neg number.
            #self.conv1.bias = torch.nn.Parameter(torch.tensor([1-self.epsilon, -(1+self.epsilon), 1-self.epsilon, -(1+self.epsilon)])) 
            self.conv1.bias = torch.nn.Parameter(torch.tensor([0]*2, dtype=torch.float))

    
    def forward(self, x):
        conv_result = self.conv1(x)

        tri = conv_result[:,0]
        sqr = conv_result[:,1]
        # max pool all pixels into 1, also remove one of the 1 dimensions
        triangle_indicator = F.max_pool2d(tri, (6,6)).squeeze(dim=1)
        square_indicator = F.max_pool2d(sqr, (6,6)).squeeze(dim=1)

        return torch.cat([triangle_indicator, square_indicator], dim=1) 


    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass


def make_DS():
    '''make the triangle square toy DS.
    1 channel 0/1 image of size 10x10. 
    only containing one of the patterns which can be shifted inside the image frame !

    
    '''

    canvas = torch.zeros( 10,10, dtype=torch.float)

    triangle = torch.tensor([ 
                [0, 1, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1]
            ])

    square = torch.tensor([
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]
            ])


    ds_list = []
    # move the images to all the 6 locations
    for i in range(6):
        for j in range(6):
            canvas = torch.zeros( 10,10, dtype=torch.float)
            canvas[i:i+5, j:j+5] = triangle
            # add a dim for batch concat
            canvas = torch.unsqueeze(canvas ,0)
            ds_list.append(canvas)

            canvas = torch.zeros( 10,10, dtype=torch.float)
            canvas[i:i+5, j:j+5] = square
            canvas = torch.unsqueeze(canvas ,0)
            ds_list.append(canvas)

          

    #then put them in 1 batch
    dataset = torch.concat(ds_list, dim=0)

    # labels 0 is triangle 1 square, they alternate, count = 36
    labels = torch.tensor([0, 1]*36)
    assert labels.size(0) == dataset.size(0)


    
    
    # shuffle the images
    """ idxs = torch.randperm(dataset.size(0)) 
    dataset = dataset[idxs]
    labels = labels[idxs] """

    # add the channel dimension
    return torch.unsqueeze(dataset, dim=1), labels


class Empty_model(nn.Module):

    def __init__(self):
        super().__init__()
        # The input is assumed to be a 1 channel 0/1 image of size 10x10. only containing one of the patterns which can be shifted inside the image frame !
        # we don't need a bias !
        self.conv1 = nn.Conv2d(1, 2, 5, bias=False)


    def forward(self, x):
        conv_out = self.conv1(x)

        tri = conv_out[:,0]
        sqr = conv_out[:,1]
        
        # max pool all pixels into 1, also remove one of the 1 dimensions
        triangle_indicator = F.max_pool2d(tri, (6,6)).squeeze(dim=1)
        square_indicator = F.max_pool2d(sqr, (6,6)).squeeze(dim=1)
        

        return torch.cat([triangle_indicator, square_indicator], dim=1)  #triangle_indicator 


def make_deterministic():
    # to make everything deterministic. 
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(0, workers=True)

def evaluate_hand_made_model(epsilon):
    robust_model = optimal_projector(epsilon)
    data, labels = make_DS()

    activations = robust_model(data)
    #print('activations shape', activations.shape)
    
    pred_labels = torch.argmax(activations, dim=1)  
    #print('pred_labels shape', pred_labels.shape)

    print('accuracy: ', (labels==pred_labels).sum()/labels.size(0) )
    print('optimal model loss:', F.cross_entropy(activations, labels))

    
    # Attack the model
    # the critical epsilon value is 0.5 above that the DS gets ambigious
    # Also the pixel statistics between the two classes is balanced.
    # The optimla model suggested is not linear since we use maxpooling

    # Note: torchattacks.APGDT wont work since it can't handle 2 classes. set_mode_targeted_least_likely() doesn't work !

    # FGSM was weak so I used PGD
    #fgsm = torchattacks.FGSM(robust_model, eps=epsilon)
    #fgsm.set_mode_targeted_least_likely()
    #x_adv = fgsm(data.clone().detach(), labels.clone().detach())

    pgd = torchattacks.PGD(robust_model, eps=epsilon, alpha=epsilon/2, steps=10)
    pgd.set_mode_targeted_least_likely()
    x_adv = pgd(data, labels)

    # feed to model
    adv_activs = robust_model(x_adv)
    adv_preds = torch.argmax(adv_activs, dim=1) 

    print('Hand made model adv acc: ', (labels==adv_preds).sum()/labels.size(0) )

    """ print('adv sampels:')
    print(labels[4])
    print(x_adv[4])
    print(labels[13])
    print(x_adv[13]) """

def evaluate_std_model(epsilon):
    data, labels = make_DS()
    #train an empty model standardly
    print()
    print("training standard model...")
    std_trained_model = Empty_model()
    optim = torch.optim.Adam(std_trained_model.parameters(), lr=.1)
    epochs = 30
    for epoch in range(epochs):
        optim.zero_grad()
        preds = std_trained_model(data)

        loss_func = nn.CrossEntropyLoss() #F.binary_cross_entropy(F.sigmoid(preds.squeeze()), (1-y).float())
        loss = loss_func(preds, labels)
        loss.backward()
        optim.step()

        label_preds = torch.argmax(preds, dim=1)  #(F.sigmoid(preds.squeeze()) > 0.5).int() 
        #if epoch%20 ==0 or (epoch== epochs-1):
        #    print(f"epoch {epoch} loss: {loss.item()}  , acc: {(labels==label_preds).sum()/labels.size(0)}")

    
    # masure standardly trained model's robust accuracy
    pgd = torchattacks.PGD(std_trained_model, eps=epsilon, alpha=epsilon/2, steps=10)
    pgd.set_mode_targeted_least_likely() 
    x_adv = pgd(data, labels)
    # feed to model
    adv_activs = std_trained_model(x_adv)
    adv_preds = torch.argmax(adv_activs, dim=1) 
    print(f"For epsilon = {epsilon} , standardly trained model's adv acc: ", (labels==adv_preds).sum()/labels.size(0) )


    
    # see if any of the attack images has an out of range pixel
    if torch.any(x_adv > 1) or torch.any(x_adv < 0):
        print("yes, there are out of range adv image pixels.")
    else:
        print("no, there are no out of range adv image pixels.")

 


    """ print('adv sampels:')
    print(labels[4])
    print(x_adv[4])
    print(labels[13])
    print(x_adv[13]) """


    """ print('trained conv activations:')

    print('conv filter 1')

    print( empty_model.conv1.weight[0] ) """

    """ print('conv filter 2')
    print( empty_model.conv1.weight[1] )"""


def main():
    # the ball size
    #epsilon = float(input("choose epsilon: "))

    # to get consistent results I should call make_deterministic() before evaluating each model 
    make_deterministic()
    #evaluate_hand_made_model(epsilon)

    make_deterministic()
    #evaluate_std_model(epsilon)


    
    # adversarialy train an empty model
    empty_model = Empty_model()
    optim = torch.optim.Adam(empty_model.parameters(), lr=0.01)
    epochs = 40
    fgsm = torchattacks.FGSM(empty_model, eps=0.2)
    fgsm.set_mode_targeted_least_likely()
    
    

    for epoch in range(epochs):
        x, y = data, labels

        
        x_adv = fgsm(data, labels)

        optim.zero_grad()

        preds = empty_model(x_adv)
        loss = F.cross_entropy(preds, y)
        loss.backward()
        optim.step()

        label_preds = torch.argmax(preds, dim=1) 

        print(f"epoch {epoch} loss: {loss.item()}  , acc: {(y==label_preds).sum()/y.size(0)}")

    
    
    print('trained conv activations:')

    print('conv filter 1')

    print( empty_model.conv1.weight[0] )

    print('conv filter 2')
    print( empty_model.conv1.weight[1] )



    empty_model = Empty_model()
    fgsm = torchattacks.FGSM(empty_model, eps=0.4)
    fgsm.set_mode_targeted_least_likely()

    x_adv = fgsm(data, labels)
    
    # feed to model
    adv_activs = empty_model(x_adv)
    adv_preds = torch.argmax(adv_activs, dim=1) 

    print('Random empty model adv acc: ', (labels==adv_preds).sum()/labels.size(0) )

    
    
    
    
    


if __name__=='__main__':
    main()

    # set the env var CUBLAS_WORKSPACE_CONFIG=:4096:8 for determinism

    
    
    
    
    
    
    
    