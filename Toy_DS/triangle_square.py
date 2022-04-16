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




class optimal_projector(pl.LightningModule):
    def __init__(self, epsilon):
        super().__init__()

        # hand tune the conv layer
        with torch.no_grad():
            # The input is assumed to be a 1 channel 0/1 image of size 10x10. only containing one of the patterns which can be shifted inside the image frame !
            self.conv1 = nn.Conv2d(1, 4, 5)

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
            # I normalize the filter by number of pixels in the image
            # so if all pixels are active I get 1.

            # I need to check two inequalities with 2 different convolutional transformations (the sign and the bias are diff)
            # So for each shape I need two filters
            self.conv1.weight[0] =  torch.nn.Parameter( 
                (-1/triangle_num_pixels) * torch.tensor([ 
                    [0, 1, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1]
                ])
            )

            self.conv1.weight[1] =  torch.nn.Parameter(
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

            self.conv1.weight[2] =  torch.nn.Parameter(
                (-1/square_num_pixels) * torch.tensor([
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]
            ]) )

            self.conv1.weight[3] =  torch.nn.Parameter(
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
            self.conv1.bias = torch.nn.Parameter(torch.tensor([0]*4, dtype=torch.float))

            



    """ def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        # the output of a 10x10 image with 5x5 conv is 6x6 (no padding).
        # The output channel [0 and 1] and [2 and 3] need to be subtracted 
        # Do the subtraction for each class by hand ! (easier)
        diff_triangle = -x[:,0] - x[:,1]
        diff_square = -x[:,2] - x[:,3]

        # max pool all pixels into 1
        triangle_indicator = F.max_pool2d(diff_triangle, (6,6)) 
        square_indicator = F.max_pool2d(diff_square, (6,6)) 

        # if the output of one is 0 then that class is the answer. otherwise the output is negative. 
        # we can also juts use softmax logic of whichever is bigger is the answer
        return torch.tensor([triangle_indicator, square_indicator]) """

    
    def forward(self, x):
        x = self.conv1(x)

        tri = x[:,1]
        sqr = x[:,3]
        # max pool all pixels into 1, also remove one of the 1 dimensions
        triangle_indicator = F.max_pool2d(tri, (6,6)).squeeze(dim=1)
        square_indicator = F.max_pool2d(sqr, (6,6)).squeeze(dim=1)

        print('tri indicator shape', triangle_indicator.shape) 

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

    # labels 0 is triangle 1 square alternate, count =36
    labels = torch.tensor([0, 1]*36)
    assert labels.size(0) == dataset.size(0)

    
    # shuffle the images
    idxs = torch.randperm(dataset.size(0)) 
    dataset = dataset[idxs]
    labels = labels[idxs]

    # add the channel dimension
    return torch.unsqueeze(dataset, dim=1), labels


class Empty_model(nn.Module):

    def __init__(self):
        super().__init__()
        # The input is assumed to be a 1 channel 0/1 image of size 10x10. only containing one of the patterns which can be shifted inside the image frame !
        # we don't need a bias !
        self.conv1 = nn.Conv2d(1, 1, 5, bias=False)


    def forward(self, x):
        x = self.conv1(x)

        tri = x[:,0]
        #sqr = x[:,1]
        
        # max pool all pixels into 1, also remove one of the 1 dimensions
        triangle_indicator = F.max_pool2d(tri, (6,6)).squeeze(dim=1)
        #square_indicator = F.max_pool2d(sqr, (6,6)).squeeze(dim=1)
        

        return triangle_indicator #torch.cat([triangle_indicator, square_indicator], dim=1) 


def main():
    
    # to make everything deterministic. 
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(0, workers=True)
    #trainer = Trainer(deterministic=True) 


    model = optimal_projector(epsilon=0.1)

    data, labels = make_DS()

    

    activations = model(data)
    print('activations shape', activations.shape)
    
    pred_labels = torch.argmax(activations, dim=1)  

    print('pred_labels shape', pred_labels.shape)

    print('accuracy: ', (labels==pred_labels).sum()/labels.size(0) )

    print('optimal model loss:', F.cross_entropy(activations, labels))

    
    # Attack the model
    # the critical epsilon value is 8/23 abive that the DS gets ambigious
    # Also the dataset statistics between the two class is balanced.
    # The optimla model suggested is not linear since we use maxpooling
    epsilon = 0.5
    import torchattacks
    """ from autoattack import AutoAttack
    adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='standard') """

    #targeted_Apgd = torchattacks.APGDT(model, n_classes=2, norm='Linf', eps = epsilon, steps=100)
    
    fgsm = torchattacks.FGSM(model, eps=epsilon)
    fgsm.set_mode_targeted_least_likely()

    x_adv = fgsm(data, labels)

    print('adv sampels:')
    print(labels[4])
    print(x_adv[4])
    print(labels[13])
    print(x_adv[13])

    # feed to model
    adv_activs = model(x_adv)
    adv_preds = torch.argmax(adv_activs, dim=1) 

    print('adv acc: ', (labels==adv_preds).sum()/labels.size(0) )


    #train an empty model
    empty_model = Empty_model()
    optim = torch.optim.Adam(empty_model.parameters(), lr=.1)
    epochs = 200
    for epoch in range(epochs):
        x, y = data, labels

        optim.zero_grad()

        preds = empty_model(x)
        loss = F.binary_cross_entropy(F.sigmoid(preds.squeeze()), (1-y).float())
        loss.backward()
        optim.step()

        label_preds = (F.sigmoid(preds.squeeze()) > 0.5).int() #torch.argmax(preds, dim=1) 

        print(f"epoch {epoch} loss: {loss.item()}  , acc: {((1-y)==label_preds).sum()/y.size(0)}")

    
    
    print('trained conv activations:')

    print('conv filter 1')

    print( empty_model.conv1.weight[0] )

    """ print('conv filter 2')
    print( empty_model.conv1.weight[1] )

 """
    """ # adversarialy train an empty model
    empty_model = Empty_model()
    optim = torch.optim.Adam(empty_model.parameters(), lr=0.01)
    epochs = 40
    fgsm = torchattacks.FGSM(empty_model, eps=0.1)
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
 """
    
    
    
    
    


if __name__=='__main__':
    main()

    # set the env var CUBLAS_WORKSPACE_CONFIG=:4096:8 for determinism

    
    
    
    
    
    
    
    """ DataLoader

    DataLoader will reseed workers following Randomness in multi-process data loading algorithm. Use worker_init_fn() and generator to preserve reproducibility:

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    ) """
