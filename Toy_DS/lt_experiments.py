from unicodedata import decimal
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

from make_T_L_dataset import make_DS

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
        """ self.data_mean = data_mean #torch.nn.parameter.Parameter(data_mean, requires_grad=False) 
        self.data_std = data_std #torch.nn.parameter.Parameter(data_std, requires_grad=False)  """


    def forward(self, input):

        """ # normalize
        input_centered = input - self.data_mean
        x = input_centered / self.data_std  """

        x = self.conv1(input)
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
        '''

        x = self.conv1(input)
        # 2 pattern filter results
        L = x[:,0]
        T = x[:,1]

        
        # max pool all pixels into 1 value, also remove one of the 1 dimensions
        L_results = F.max_pool2d(L, (2,2), return_indices=True)
        T_results = F.max_pool2d(T, (2,2), return_indices=True)
        L_indicator, L_index = L_results[0].squeeze(dim=1), L_results[1].squeeze()
        T_indicator, T_index = T_results[0].squeeze(dim=1), T_results[1].squeeze()


        return torch.cat([L_indicator, T_indicator], dim=1), L_index, T_index

def advresarial_evaluation(std_trained_model, std_data, std_labels, attack_epsilon, alpha_ratio=0.5, steps=10, inference=False, absolute_alpha=None):
    # masure standardly trained model's robust accuracy
    if absolute_alpha is None:
        print("using alpha relative to size of epsilon.")
        pgd = torchattacks.PGD(std_trained_model, eps=attack_epsilon, alpha=attack_epsilon*alpha_ratio, steps=steps)
    else:
        print("using absolute alpha.")
        pgd = torchattacks.PGD(std_trained_model, eps=attack_epsilon, alpha=absolute_alpha, steps=steps)

    pgd.set_mode_targeted_least_likely() 
    x_adv = pgd(std_data, std_labels)

    with torch.no_grad():
        L_index = T_index = -1
        if inference:
            adv_activs, L_index, T_index = std_trained_model.predict(x_adv)
            adv_preds = torch.argmax(adv_activs, dim=1) 
        else:
        # feed to model
            adv_activs = std_trained_model(x_adv)
            adv_preds = torch.argmax(adv_activs, dim=1) 

    return adv_preds, x_adv, L_index, T_index
    


def standard_training(checkpoint_path, epochs, lr, dataloader, mean, std, gpu_id, eval_frequency=5):
    '''
    checkpoint_path: path of the model checkpoint, if it exists it will be loaded instead of training.
    '''
    try:
        
        std_trained_model = Empty_model(pattern_size=3, data_mean=mean, data_std=std)
        std_trained_model.load_state_dict(torch.load(checkpoint_path))
        std_trained_model.eval()
        print()
        print("loaded model from disk. \n")

    except FileNotFoundError: 
        print()
        print(f"training standard model, epochs={epochs}, lr={lr} \n")

        device = torch.device('cuda:'+gpu_id)
        std_trained_model = Empty_model(pattern_size=3, data_mean=mean, data_std=std)
        std_trained_model.to(device)
        
        optim = torch.optim.Adam(std_trained_model.parameters(), lr=lr)

        for epoch in tqdm(range(epochs)):
            for idx, data in tqdm(enumerate(dataloader)):
                #as defined in our collate function
                images, labels = data[0].to(device), data[1].to(device)

                #print("images shape: ", images.shape)
                #print("images GPU mem size (byte)", images.element_size() * images.nelement())

                optim.zero_grad()
                preds = std_trained_model(images)

                loss_func = nn.CrossEntropyLoss()
                loss = loss_func(preds, labels)
                loss.backward()
                optim.step()

                

            if (epoch+1)% eval_frequency ==0 or (epoch== epochs-1):
                print() 
                print("evaluating ...")
                with torch.no_grad():
                    # print report on the whole dataset
                    total=0
                    correct=0
                    loss=0
                    for idx, data in tqdm(enumerate(dataloader)):
                        images, labels = data[0].to(device), data[1].to(device)
                        preds = std_trained_model(images)
                        label_preds = torch.argmax(preds, dim=1)

                        total += labels.size(0)
                        correct += (labels==label_preds).sum().item()

                        loss_func = nn.CrossEntropyLoss()
                        loss += loss_func(preds, labels)
                    
                    print(f"epoch {epoch} loss: {loss.item()}") 
                    print(f"acc: {correct/total}") 
                    print(f"incorrectly classified: {total-correct}")
        
        torch.save(std_trained_model.state_dict(), checkpoint_path)

    return std_trained_model

def loader(img_path):
        # cast each group of images from 8bits to float
        scaling = 100
        centering = 0.5
        image_group = torch.load(img_path).float() /scaling + centering
        return image_group.round(decimals=6) #floating point can have inaccuracies so I round

def L_T_collate(list_of_groups_of_images_and_labels):
        # each "sample" given to collate is: (group of images, single label) 
        # we just need to concatinate images to get a single batch
        # and replicate the labels into size of groups and batch them
        images = []
        labels = []
        for (grouped_images, single_label) in list_of_groups_of_images_and_labels:
            images.append(grouped_images)
            labels.append( torch.full((grouped_images.shape[0],), single_label) ) # making them tensor here makes it faster

        final_batched_images = torch.concat(images, dim=0)
        final_batched_labels = torch.concat(labels)

        # shuffle the images
        idxs = torch.randperm(final_batched_images.size(0)) 
        final_batched_images = final_batched_images[idxs]
        final_batched_labels = final_batched_labels[idxs]

        # insert channel dimension for images
        return (final_batched_images.unsqueeze(dim=1), final_batched_labels)

if __name__ == '__main__':

    '''
    Important note: 
    This code can raise: 
    
    "RuntimeError: kthvalue CUDA does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'. 
    You can turn off determinism just for this operation, or you can use the 'warn_only=True' option, if that's acceptable for your application. 
    You can also file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation."

    when training a model for the first time and when the model is not loaded from memory. 
    But runing the script again resolves the problem (I guess since the model is loaded from memory ?! and that apparently makes the model deterministic ??) 


    '''

    # to make everything deterministic. 
    make_deterministic()

    # make dataset and loader, we need this since all the images are 23Gibs (each pixel a 32bit float) on the GPU and we get an overflow
    ds_path = "./L_T_dataset_groupsize_729_include_boundaryTrue_epsilon0.41_center_0.5_scale_100"


    L_T_dataset = DatasetFolder(ds_path, loader=loader, extensions='.pt')
    print("dataset info: ")
    print(L_T_dataset.class_to_idx)
    print("dataset len: ", len(L_T_dataset))

    # dataloader
    batch_size = 170
    num_workers= 16
    L_T_dataloader = DataLoader(L_T_dataset, batch_size=batch_size, shuffle=True, num_workers= num_workers, collate_fn=L_T_collate, pin_memory=True)
    epochs = 15
    lr= 0.1
    checkpoint_path = './LT_checkpoints/robust_model_trained_on_groupsize_729_includes_std_images.pt'
    try:
        std_trained_model = standard_training(checkpoint_path,epochs, lr, L_T_dataloader, mean=0., std=1., gpu_id='5', eval_frequency=5)
    except RuntimeError as err: 
        print(f"error is: {err}, {type(err)=}")
        print("if it is the Unable to find a valid cuDNN algorithm to run convolution error,\nit is due to memory reduce batch size or group size (remake the dataset).")

    
    # note the dataset and attack epsilons are different since we should have data_epsilon > attack_epsilon
    # PGD attack params
    attack_epsilons = np.arange(0.38, 0.51, 0.01)
    alpha_ratio = 0.1
    steps = 50
    print(f"PGD ( attack_epsilon={attack_epsilons}, alpha_ratio={alpha_ratio}, steps={steps} )")
    # I need to fetch a dataset that doesn't include the boundary samples for advresarial evaluation.
    std_images, std_labels, _, _ = make_DS(include_epsilon_bounadry=False)
    std_images = std_images.unsqueeze(dim=1)

    robustness_results = []
    for atk_eps in attack_epsilons:
        adv_preds, adv_images, L_indexes, T_indexes = advresarial_evaluation(
                std_trained_model, std_images, std_labels,
                atk_eps, alpha_ratio, steps, inference=True, absolute_alpha=0.4*0.1)
                
        
        robust_acc = (std_labels==adv_preds).sum()/std_labels.size(0)
        num_misclassified = (std_labels!=adv_preds).sum()
        robustness_results.append((atk_eps, robust_acc.item(), num_misclassified.item()))

        print("acc: ", robust_acc)


    
    print("adv results")
    print(robustness_results)
    
    
    print("conv filters:")
    print()

    print("L shape(0)")
    print(std_trained_model.conv1.weight[0]) 

    print("T shape(1)")
    print(std_trained_model.conv1.weight[1]) 
    

    """ 
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
            print(f"L_index: {L_index}, T_index: {T_index}") """




"""
# check dataset
 print("images shape: ", images.shape)
            print("labels shape: ", labels.shape)

            print('count of each class:')
            unique, counts = np.unique(labels, return_counts=True)
            print(dict(zip(unique, counts)))

            print('distinc pixel values:')
            unique, counts = np.unique(images.numpy().flatten(), return_counts=True)
            print(dict(zip(unique, counts)))
"""



