import torch
from torch import nn
import torch.nn.functional as F

import random
import numpy as np


import pytorch_lightning as pl
from pytorch_lightning import seed_everything

import torchattacks


class Empty_model(nn.Module):

    def __init__(self):
        super().__init__()
        # The input is assumed to be a 1 channel 0/1 image of size 3x3. only containing one of the patterns which can be shifted inside the image frame !
        # we don't need a bias !
        self.conv1 = nn.Conv2d(1, 3, 2, bias=False)


    def forward(self, x):
        x = self.conv1(x)

        slant = x[:,0]
        horizontal = x[:,1]
        vertical = x[:,2]
        
        # max pool all pixels into 1 value, also remove one of the 1 dimensions
        slant_indicator = F.max_pool2d(slant, (2,2)).squeeze(dim=1)
        horizontal_indicator = F.max_pool2d(horizontal, (2,2)).squeeze(dim=1)
        vertical_indicator = F.max_pool2d(vertical, (2,2)).squeeze(dim=1)
        

        return torch.cat([slant_indicator, horizontal_indicator, vertical_indicator], dim=1)


def make_DS(disc_intervals, epsilon_intervals):
    ''' This toy dataset was too simplistic (only two active pixels per pettern) 
    and training the model standardly gave an almost robust model.
    The idea is that we need to have a set of patterns such that 
    the model can differentiate them only using a subset of the pattern not the whole shape.
    Like in the triangle-square dataset.  

    All images have 1 channel, the standard starting images only have 0 or 1 pixels.  
    All possible images of size 3x3 that are included in the epsilon ball of each standard sample will be included.
    This will be all potential advresarial examples since: 
        1. The standard data distribution is finite (only 12 standard images) 
        2. I manually set the pixel precision to a low value (pixels are discretized !)

    Each image only contains one of the patterns which can be shifted inside the canvas !

    '''

    

    slanting = torch.tensor([ 
                [1, 0],
                [0, 1]
            ])

    horizontal = torch.tensor([
                [1, 1],
                [0, 0]
            ])

    vertical = torch.tensor([
                [1, 0],
                [1, 0]
            ])


    dataset_list = []
    # move the images to all the 4 locations
    for i in range(2):
        for j in range(2):
            canvas = torch.zeros(3, 3, dtype=torch.float)
            canvas[i:i+2, j:j+2] = slanting
            # add a dim for batch concat
            canvas = torch.unsqueeze(canvas ,0)
            dataset_list.append(canvas)

            canvas = torch.zeros(3, 3, dtype=torch.float)
            canvas[i:i+2, j:j+2] = horizontal
            canvas = torch.unsqueeze(canvas ,0)
            dataset_list.append(canvas)

            canvas = torch.zeros(3, 3, dtype=torch.float)
            canvas[i:i+2, j:j+2] = vertical
            canvas = torch.unsqueeze(canvas ,0)
            dataset_list.append(canvas)


    #then put them in 1 batch
    clean_dataset = torch.concat(dataset_list, dim=0)

    # labels 0 is slanting, 1 horizontal, 2 vertical. they alternate, count =4
    clean_labels = torch.tensor([0, 1, 2]*4)
    assert clean_labels.size(0) == clean_dataset.size(0)

    
    # shuffle the images
    idxs = torch.randperm(clean_dataset.size(0)) 
    clean_dataset = clean_dataset[idxs]
    clean_labels = clean_labels[idxs]

    # add the channel dimension
    return torch.unsqueeze(clean_dataset, dim=1), clean_labels


# remmember to namralize the dataset (specially for the attacks) since now the pixels are no longer between 0,1 !

if __name__ == '__main__':

    # to make everything deterministic. 
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(0, workers=True)



    disc_intervals = 6
    epsilon = 2
    data, labels = make_DS(disc_intervals, epsilon_intervals=epsilon)

    """ data = torch.squeeze(data)

    print(data.shape)
    print(labels.shape)

    for (img, label) in zip(data, labels):
        print()
        print(label)
        print(img) """

    
    print()
    print("training standard model:")
    std_trained_model = Empty_model()
    optim = torch.optim.Adam(std_trained_model.parameters(), lr=.1)
    epochs = 200
    for epoch in range(epochs):
        # data, labels

        optim.zero_grad()

        preds = std_trained_model(data)

        loss_func = nn.CrossEntropyLoss() #F.binary_cross_entropy(F.sigmoid(preds.squeeze()), (1-y).float())
        loss = loss_func(preds, labels)
        loss.backward()
        optim.step()

        

        if epoch%20 ==0:
            # print report
            label_preds = torch.argmax(preds, dim=1)  #(F.sigmoid(preds.squeeze()) > 0.5).int() 
            print(f"epoch {epoch} loss: {loss.item()}  , acc: {(labels==label_preds).sum()/labels.size(0)}")

            """ print('preds')
            print(preds.size())

            print('labels')
            print(labels.size()) """

    epsilon = 0.4
    fgsm = torchattacks.FGSM(std_trained_model, eps=epsilon)
    fgsm.set_mode_targeted_least_likely()

    x_adv = fgsm(data, labels)

    print('adv sampels:')
    print(labels[4])
    print(x_adv[4])
    print(labels[10])
    print(x_adv[10])

    # feed to model
    adv_activs = std_trained_model(x_adv)
    adv_preds = torch.argmax(adv_activs, dim=1) 

    print("standardly trained model's adv acc: ", (labels==adv_preds).sum()/labels.size(0) )


    print("conv filters:")
    print()

    print("slanting(0)")
    print(std_trained_model.conv1.weight[0]) 

    print("horizontal(1)")
    print(std_trained_model.conv1.weight[1]) 

    print("vertical(0)")
    print(std_trained_model.conv1.weight[2])    