import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import attr
import logging as log
import pytorch_lightning as pl
from typing import List
from tqdm import tqdm


#from Dataset_measure import concentration_measure 
import numpy as np

from .Measurements import Measure


@attr.s
class Adversarisal_bench:
    '''the main class that the user is going to interact with '''
    
    # Takes pretrained model. The model should have the forward method implemented
    model:nn.Module = attr.ib()
    # the function that transforms the output of the network into labels. takes inputs in batches.
    predictor = attr.ib()
    device = attr.ib(default='cuda:0')

    def __attrs_post_init__(self):
        log.info(f'device is: {self.device}')
        # we don't change the weights of the model
        self.model.eval().to(self.device)
        


    def measure(self, whole_dataset:pl.LightningDataModule, measures:List[Measure], attacks, on_train=True, on_val=True, on_test=True):
        ''' whole_dataset: is a lightning data module that includes data for train, val and test.
            If one is not needed None can be sent and None willl be returned.
            
            on_[dataset_subset]: a booleian variable indicating which part the data to use.
            
            
            returns the results as a list [train_results, val_results, test_results] each one a list of [result_of_on_clean, result_of_on_attack].
            if a measure doesn't implement on_clean or on_attack returns None as the result. 
        '''

        train_results = val_results = test_results = None
        if on_train:
            print('Measuring on Train set:')
            train_results = self._measure(whole_dataset.train_dataloader(), measures, attacks)
        if on_val:
            print('Measuring on Validation set:')
            val_results = self._measure(whole_dataset.val_dataloader(), measures, attacks)
        if on_test:
            print('Measuring on Test set:')
            test_results = self._measure(whole_dataset.test_dataloader(), measures, attacks)

        return train_results, val_results, test_results

    def _measure(self, dataloader:DataLoader, measures:List[Measure], attacks):
        ''' Calculates the results of measures on the data.
            calls each measure's on_clean on each batch. 
            calls each measure's on_attack for each attack and on each batch .
        '''
        if dataloader is None:
            return None

        for data in tqdm(dataloader):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            outputs = self.model(inputs)
            predicted_labels = self.predictor(outputs)

            # calculate measures that just need the clean data
            for m in measures:
                m.on_clean_data(self.model, inputs, labels, outputs, predicted_labels)

            #save old model state
            old_state = self.model.state_dict()

            # calculate the measures that are based on an attack
            for attack in attacks:
                # make adversarial images 
                adv_inputs = attack(inputs, labels) # the model should be already sent to init the attack (according to torchattacks)
                adv_outputs = self.model(adv_inputs)
                adv_predictions = self.predictor(adv_outputs)
                for m in measures:
                    m.on_attack_data(self.model, inputs, labels, outputs, predicted_labels, adv_inputs, adv_outputs, adv_predictions, attack)
            
            #finilize the result for each batch
            for m in measures:
                m.batch_end()
            
        # get results for the whole dataset
        results = []
        for idx, m in enumerate(measures):
            if idx ==0:
                print('normal acc stats')
                print(m._total)
                print(m._correct)
            results.append(m.final_result())

        ########################
        # Debug
        #check if model is changed by the attack
        """ def compare_models(dict1, model_2):
            models_differ = 0
            for key_item_1, key_item_2 in zip(dict1.items(), model_2.state_dict().items()):
                if torch.equal(key_item_1[1], key_item_2[1]):
                    pass
                else:
                    models_differ += 1
                    if (key_item_1[0] == key_item_2[0]):
                        print('Mismtach found at', key_item_1[0])
                    else:
                        raise Exception
            if models_differ == 0:
                print('Models match perfectly! :)')

        compare_models(old_state, self.model) """
        

        return results




    
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







