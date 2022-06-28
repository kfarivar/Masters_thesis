import torch
from torch import cartesian_prod, nn
import torch.nn.functional as F

import random
import numpy as np
from tqdm import tqdm 


import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder

import torchattacks

def make_deterministic():
    # to make everything deterministic. 
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(0, workers=True)


class Empty_model(nn.Module):

    def __init__(self, pattern_size, data_mean, data_std):
        super().__init__()
        '''
        data-mean/std: used to normalize the data.
        '''
        # The input is assumed to be a 1 channel 0/1 image of size canvas_size*canvas_size. only containing one of the patterns which can be shifted inside the image frame !
        # we don't need a bias !
        self.conv1 = nn.Conv2d(1, 2, pattern_size, bias=False)
        self.patter_size = pattern_size
        self.data_mean = torch.nn.parameter.Parameter(data_mean, requires_grad=False) 
        self.data_std = torch.nn.parameter.Parameter(data_std, requires_grad=False) 


    def forward(self, input):
        '''
        get_max_index: whether to return the max index for maxpooling.
        '''
        # normalize
        input_centered = input - self.data_mean
        x = input_centered / self.data_std 

        x = self.conv1(x)
        # 2 pattern filter results
        L = x[:,0]
        T = x[:,1]
        
        # max pool all pixels into 1 value, also remove one of the 1 dimensions
        L_indicator = F.max_pool2d(L, (2,2)).squeeze(dim=1)
        T_indicator = F.max_pool2d(T, (2,2)).squeeze(dim=1)

        return torch.cat([L_indicator, T_indicator], dim=1)

    def predict(self, input):
        '''
        used only for inference never for training
        Always returns the index of maximum maxpool element.
        The index is a single value and the matrix is flattened row major then indexed.

        get_max_index: whether to return the max index for maxpooling.
        '''
        # normalize
        input_centered = input - self.data_mean
        x = input_centered / self.data_std 

        x = self.conv1(x)
        # 2 pattern filter results
        L = x[:,0]
        T = x[:,1]

        
        # max pool all pixels into 1 value, also remove one of the 1 dimensions
        L_results = F.max_pool2d(L, (2,2), return_indices=True)
        T_results = F.max_pool2d(T, (2,2), return_indices=True)
        L_indicator, L_index = L_results[0].squeeze(dim=1), L_results[1].squeeze()
        T_indicator, T_index = T_results[0].squeeze(dim=1), T_results[1].squeeze()


        return torch.cat([L_indicator, T_indicator], dim=1), L_index, T_index
        
        



def make_DS(include_epsilon_bounadry=False, epsilon=0.4, random_sample_boundary=False, calculate_normalization=0, symmetric_options=True, perturb_pattern_pixels_only=False):  
    '''make the boundary L vs T pattern toy DS.
    The canvas is 4x4 and the patterns 3x3.

    For each standard image (if include_epsilon_bounadry==True) I include all the images that can be created using the advresarial perturbation,
    I will wither apply no change to a pixel xor +epsilon if it is 0 xor -epsilon if it is 1. 
    (keep the potential davresarial examples in the [0,1] bound due to my intuition.)


    include_epsilon_bounadry: whether to include extra epsilon ball boundary samples in the dataset according to l-infinity norm. 
    epsilon: the advresarial perturbation size for l-infinity.
    random_sample_boundary: (include_boundary must be True to be used) 
        False: use either 0 or epsilon when choosing pixel options
        True: use either 0 or a random perturbation samples uniformly from (0,epsilon]

    calculate_normalization: whether to calcaulate normalization factors.
        ==1: normalize pixels using all pixels mean and std
        ==2: normalize each pixel using the corresponding pixels in other images max and min
    
    symmetric_options: 
        True: include both -epsilon and +epsilon for both 0 and 1 pixels (in addition to 0)
        False: include only +epsilon for 0 pixels and only -epsilon for 1 pixels. (in addition to 0)

    perturb_pattern_pixels_only:
        True: only perturb the pixels in the 3x3 pattern and not the whole 4x4 canvas
        False: perturb all pixels on the canvas

    Each image only contains one of the patterns which can be shifted inside the canvas !

    '''

    # the size of one side of the square canvas.
    canvas_size = 4
    # the size of one side of the square pattern.
    pattern_size = 3

    

    L_pattern = torch.tensor([ 
                [1, 0, 0],
                [1, 0, 0],
                [1, 1, 1]
            ],
            dtype=torch.float32 )

    T_pattern = torch.tensor([
                [1, 1, 1],
                [0, 1, 0],
                [0, 1, 0]
            ],
            dtype=torch.float32 )

    # The patterna and labels should match one to one
    # label 0 is L_pattern, 1 T_pattern
    std_patterns = [L_pattern, T_pattern]
    std_pattern_labels = [0,1]

    if perturb_pattern_pixels_only and include_epsilon_bounadry:
        print("perturbing only pattern pixels: ")    
        # add the potential adv patterns to standard patterns
        # Assume the image is flattened, 
        # I first create a list of set of options for each pixel: if pixel==0: {0,+epsilon} elif pixel==1: {0,-epsilon} else: error
        # then I take the cartisian product of all the sets and reshape it back to the shape of original image
        # then I add the original image to the created pattern. 
        # In this process the standard patterns are also created so no need to add them.
        patterns= []
        pattern_labels = []
        for std_pattern, std_label in zip(std_patterns, std_pattern_labels):

            if random_sample_boundary:
                # randomly choose a value in [0, epsilon) and use it instead of epsilon in the options
                # To make the comparison more fair I sample from [0.9*epsilon, epsilon)
                random_boundary = np.random.default_rng().uniform(low=0.8*epsilon, high=epsilon)
                print("random boundary is: ", random_boundary)
                list_of_options = [torch.tensor([0, random_boundary]) if pixel==0.0 else torch.tensor([0, -random_boundary]) for pixel in std_pattern.flatten()]

            else:
                print("deterministic epsilon = : ", epsilon)
                if symmetric_options: 
                    print("symmetric neighbours:")
                    list_of_options = [torch.tensor([0.0, epsilon, -epsilon]) if pixel==0.0 else torch.tensor([0.0, -epsilon, epsilon]) for pixel in std_pattern.flatten()]
                else: 
                    list_of_options = [torch.tensor([0.0, epsilon]) if pixel==0.0 else torch.tensor([0.0, -epsilon]) for pixel in std_pattern.flatten()]

            cartesian_of_options = cartesian_prod(*list_of_options)

            for option in cartesian_of_options:
                adv_image = option.reshape(std_pattern.shape[0], std_pattern.shape[1])
                adv_pattern = std_pattern + adv_image
                patterns.append(adv_pattern)
                pattern_labels.append(std_label)


    else: 
        patterns = std_patterns
        pattern_labels = std_pattern_labels



    dataset_list = []
    final_labels = []
    # move the images to all the 4 locations
    for i in range(2):
        for j in range(2):
            for pattern, label in zip(patterns, pattern_labels):
                canvas = torch.zeros(canvas_size, canvas_size, dtype=torch.float)
                canvas[i:i+pattern_size, j:j+pattern_size] = pattern

                 
                if not perturb_pattern_pixels_only and include_epsilon_bounadry and symmetric_options:
                    # if we want to perturb all canvas pixels we do it here
                    # note the standard image is one of the options here
                    print("preturbing all canvas pixels symmetrically.")
                    list_of_options = [torch.tensor([0.0, epsilon, -epsilon]) if pixel==0.0 else torch.tensor([0.0, -epsilon, epsilon]) for pixel in canvas.flatten()]
                    cartesian_of_options = cartesian_prod(*list_of_options)

                    # all possible potential advresarial images
                    all_adv_perturbations = cartesian_of_options.reshape(cartesian_of_options.shape[0], canvas_size, canvas_size)
                    # broadcast canvas addition
                    all_adv_images = canvas + all_adv_perturbations
                    dataset_list.append(all_adv_images)
                    final_labels.append( torch.full((all_adv_images.shape[0],), label) ) # making them tensor here makes it faster
                    
                    """ for option in cartesian_of_options:
                        adv_perturbs = option.reshape(canvas_size, canvas_size)
                        adv_pattern = canvas + adv_perturbs
                        dataset_list.append(torch.unsqueeze(adv_pattern, 0))
                        final_labels.append(label) """
                
                elif not perturb_pattern_pixels_only and not include_epsilon_bounadry:
                    print("only adding standard images in the dataset.")
                    # add a dim for batch concat 
                    canvas = torch.unsqueeze(canvas ,0)
                    dataset_list.append(canvas)
                    final_labels.append(torch.tensor([label]))
                
                else:
                    raise NotImplementedError("your dataset creation parameters are not implemenetd.")

                

    
    #then put them in 1 batch
    final_dataset = torch.concat(dataset_list, dim=0)
    final_labels = torch.concat(final_labels)
    assert final_labels.size(0) == final_dataset.size(0)
    

    # default values no normalization
    mean = torch.tensor([0])
    std = torch.tensor([1])
    if calculate_normalization == 1:
        mean = final_dataset.mean() 
        std = final_dataset.std()
        print(f"Data normalized with all pixels mean {mean} and std {std}")

    elif calculate_normalization == 2:
        raise NotImplementedError("normalization not implemented")
        """ print("Data normalized with each pixels corresponding min and max")
        normalized_data = final_dataset.clone()
        normalized_data = normalized_data.view(final_dataset.size(0), -1)
        normalized_data -= normalized_data.min(1, keepdim=True)[0]
        normalized_data /= normalized_data.max(1, keepdim=True)[0]
        normalized_data = normalized_data.view(final_dataset.size(0), final_dataset.size(1), final_dataset.size(2))
        final_dataset = normalized_data """

    # add the channel dimension, note that mean can be min and std can be max depending on the type of normlaization
    return torch.unsqueeze(final_dataset, dim=1), final_labels, mean, std





def advresarial_evaluation(std_trained_model, std_data, std_labels, attack_epsilon, alpha_ratio=0.5, steps=10, inference=False):
    # masure standardly trained model's robust accuracy
    pgd = torchattacks.PGD(std_trained_model, eps=attack_epsilon, alpha=attack_epsilon*alpha_ratio, steps=steps)
    pgd.set_mode_targeted_least_likely() 
    x_adv = pgd(std_data, std_labels)

    L_index = T_index = -1
    if inference:
        adv_activs, L_index, T_index = std_trained_model.predict(x_adv)
        adv_preds = torch.argmax(adv_activs, dim=1) 
    else:
    # feed to model
        adv_activs = std_trained_model(x_adv)
        adv_preds = torch.argmax(adv_activs, dim=1) 

    return adv_preds, x_adv, L_index, T_index
    

def standard_training(epochs, lr, images, labels, mean, std, gpu_id):
    print()
    print(f"training standard model, max_epochs={epochs}, lr={lr} \n")

    
    std_trained_model = Empty_model(pattern_size=3, data_mean=mean, data_std=std)
    if int(gpu_id) >= 0:
        # move model and data to gpu
        device = torch.device('cuda:'+gpu_id)
        std_trained_model.to(device)
        images = images.to(device)
        labels = labels.to(device)

    optim = torch.optim.Adam(std_trained_model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs)):
        # data, labels

        optim.zero_grad()

        preds = std_trained_model(images)

        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(preds, labels)
        loss.backward()
        optim.step()

        

        if epoch% int(epochs/5) ==0 or (epoch== epochs-1):
            # print report
            label_preds = torch.argmax(preds, dim=1) 
            print(f"epoch {epoch} loss: {loss.item()}  , acc: {(labels==label_preds).sum()/labels.size(0)}, incorrectly classified: {(labels!=label_preds).sum()}")

        """ if  (epoch== epochs-1): # or (epoch + 1)% int(epochs/2) == 0
            adv_preds = advresarial_evaluation(std_trained_model, std_data, std_labels, attack_epsilon, alpha_ratio, steps)
            print(f"\nAdvresarial evaluation results:\nstandardly trained model's adv acc: {(std_labels==adv_preds).sum()/std_labels.size(0)},\nCount of advresarial examples misclassified: {(std_labels!=adv_preds).sum()}")
        """

    return std_trained_model

if __name__ == '__main__':

    # to make everything deterministic. 
    make_deterministic()

    # note the dataset and attack epsilons are different since it should be the case that data_epsilon > attack_epsilon
    data_epsilon = 0.41
    calculate_normalization=0
    data, labels, mean, std = make_DS(include_epsilon_bounadry=True, epsilon=data_epsilon, 
                                        random_sample_boundary=False, calculate_normalization=calculate_normalization, 
                                        symmetric_options= True, perturb_pattern_pixels_only=False)
    if calculate_normalization == 1:
        print("applying the normalization to 0 and 1")
        new_0 = (0-mean)/std
        new_1 = (1-mean)/std
        print(f"0 is: {new_0}")
        print(f"1 is: {new_1}")
        print(f"threshold is: {(new_0 + new_1) /2}")

    # PGD attack params
    attack_epsilon = 0.4
    alpha_ratio = 0.1
    steps = 50
    print(f"PGD ( attack_epsilon={attack_epsilon}, alpha_ratio={alpha_ratio}, steps={steps} )")
    # I need to fetch a dataset that doesn't include the boundary samples for advresarial evaluation.
    std_data, std_labels, _, _ = make_DS(include_epsilon_bounadry=False)

    print("dataset shape")
    print(data.shape)
    print("labels shape")
    print(labels.shape)


    epochs = 400
    lr= 0.1
    std_trained_model = standard_training(epochs, lr, data, labels, mean, std, gpu_id='5')

    
    
    adv_preds, adv_images, L_indexes, T_indexes = advresarial_evaluation(std_trained_model, std_data, std_labels, attack_epsilon, alpha_ratio, steps, inference=True)
    print(f"""
        Advresarial evaluation results:  
        standardly trained model's adv acc: {(std_labels==adv_preds).sum()/std_labels.size(0)},
        Count of advresarial examples misclassified: {(std_labels!=adv_preds).sum()}
        """  )
    
    

    


    print("conv filters:")
    print()

    print("L shape(0)")
    print(std_trained_model.conv1.weight[0]) 

    print("T shape(1)")
    print(std_trained_model.conv1.weight[1]) 

    # print all incorrectly classified adv images
    print('all incorrectly classified adv images:')
    for img, adv_img, label, pred, L_index, T_index in zip(std_data.squeeze(), adv_images.squeeze(), std_labels, adv_preds, L_indexes, T_indexes):
        if pred != label:
            print()
            print(f"label: {label}")
            print("original")
            print(img)
            print("advresarial")
            print(adv_img)
            print(f"L_index: {L_index}, T_index: {T_index}")



    '''
    The discritization idea:
    disc_intervals, epsilon_intervals

    disc_intervals: 
            the number of segments to divide the [0,1] interval.
            I will include all possible images inside the epsilon ball using the discrete values in [0,1] AND
            the pixel values below 0 and above 1 that are inside the epsilon ball boundary in the dataset. 

    epsilon_intervals: 
        this is a discrete version of epsilon incidating radius 
        using the number of discrete intervals to include in it rather than an absolute value !

    All images have 1 channel, the standard starting images only have 0 or 1 pixels.  
    All possible images of size 4x4 that are included in the epsilon ball of each standard sample will be included.
    This will be all potential advresarial examples since: 
        1. The standard data distribution is finite (only 8 standard images) 
        2. I manually set the pixel precision to a low value (pixels are discretized !)


    # remmember to namralize the dataset (specially for the attacks) since now the pixels are no longer between 0,1 !
    '''


    """ # Check if the dataset is correct
    indexes = np.random.choice(range(data.shape[0]), size=10)
    for i in indexes:
        print(labels[i])
        print(data[i])


    # see if any of the attack images has an out of range pixel
    if torch.any( data > 1) or torch.any(data < 0):
        print("yes, there are out of range adv image pixels.")
    else:
        print("no, there are no out of range adv image pixels.")

    print('distinc pixel values:')
    unique, counts = np.unique(data.numpy().flatten(), return_counts=True)
    print(dict(zip(unique, counts)))

    print('count of each class:')
    unique, counts = np.unique(labels, return_counts=True)
    print(dict(zip(unique, counts))) """