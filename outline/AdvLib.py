import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader

from Attacks import Attack
from Dataset_measure import concentration_measure 
import numpy as np

from Measurements import measure


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

    def measure(whole_dataset:(DataLoader,DataLoader), measures:List(measure)):
        ''' whole_dataset is a tuple of (train_dataloader, test_dataloader) the masure chooses which one to use
            calls on_clean and on_attack for each measure.
            returns the results as tupels [on_clean result, on_attack result] in the same order as given.
            if a method doesn't apply returns None as the result. 
        '''
        all_results = []
        with torch.no_grad():
            for m in measures:
                clean_result = m.on_clean_data(whole_dataset)
                attack_result = m.on_attack_data(whole_dataset)
                all_results.append([clean_result, attack_result])

        return all_results


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







