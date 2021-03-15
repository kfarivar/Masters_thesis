import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader

from Attacks import Attack
from Dataset_measure import concentration_measure 
import numpy as np

class Adversarisal_bench:
    '''the main class that the user is going to interact with '''

    def __init__(self, model:nn.Module, use_cuda:bool):
        '''take the pretrained model and whether to use a gpu
           The model should have the forward method implemented'''
        device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu") 
        self.model = model.to(device)
        print("device is:" + device)
        # we don't change the weights of the model
        self.model.eval()

    def measure_models_robust_accuracy(test_set:DataLoader, attacks:list[Attack]):
        '''test_set: is a torch.utils.data.Dataloader. 
           usually implemented using torch.utils.data.Dataset (https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
           The advantage is we can use Dataparallel to speed up the process up (https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html?highlight=dataparallel)
           
           attacks: a list of attacks to perform on each test sample  
        '''
        epsilon = 0.001
        correct = 0
        adverserial_examples = []
        #save the number of corrects for each attack (number of failed attacks)
        failed_attacks = np.zeros(len(attacks))
        # assuming batch size is 1
        for batch_idx, batch in enumerate(test_set):
            data, label = batch

            # predict
            output = self.model(data)
            #forget about the examples that are already wrong
            if output != label:
                continue

            # iterate through all attacks and save results
            for attack_idx, attack_method in enumerate(attacks):
                # result will be the distorted sample and the new class
                result = attack_method.attack(data,label,target)
                attack_name = type(attack_method).__name__
                # check the result
                if result is None:
                    # print the failed attack
                    print(attack_name + "attack failed")
                    failed_attacks[attack_idx]+=1
                else :
                    # add the attack result to a list
                    adverserial_examples.append((attack_name, data, label, result))

        # return percentage of attacks that didn't succeed and the samples that did fool
        return (failed_attacks/dataloader.size(), adverserial_examples)  


    def measure_dataset_concentration(whole_dataset:DataLoader):
        return concentration_measure(whole_dataset)

    
    def make_non_robust_dataset(train_set, attack_method):
        ''' Make a non-robust train set
        '''
        # map each target to another class
        target_map = {}
        for batch_idx, batch in enumerate(train_set):
            data, label = batch

            new_sample = attack_method.attack(sample, label, target_map[label])

            # very similar to measure_models_robust_accuracy the main difference is that having targets 
            # here is mandatory. 
        
        return # new dataset

    def make_robust_dataset(train_set):
        pass







