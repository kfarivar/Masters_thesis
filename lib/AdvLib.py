import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import attr
import logging as log
import pytorch_lightning as pl
from typing import List
from tqdm import tqdm
from tqdm import trange


#from Dataset_measure import concentration_measure 
import numpy as np

from .Measurements import Measure
from .Trainer import Trainer


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


        
    def train_val_test(self, trainer:Trainer, num_epochs:int, whole_dataset:pl.LightningDataModule,measures:List[Measure], 
                        attacks, train_measure_frequency=100, val_measure_frequency=100):
        ''' Uses 'robustly_train' function to train and validate and 'evaluate_measures' to test.
            Evaluates on training/validation set with '[train/val]_measure_frequency' (if epoch_index % measure_frequency == 0)
        '''
        print('Training:')
        train_val_result = self.robustly_train(trainer, num_epochs, whole_dataset, measures, attacks, 
                                                train_measure_frequency, val_measure_frequency)

        print('Testing:')
        test_result =  self.evaluate_measures(whole_dataset.test_dataloader(), measures, attacks)

        return train_val_result, test_result


    def measure_splits(self, whole_dataset:pl.LightningDataModule, measures:List[Measure], attacks, 
                on_train=True, on_val=True, on_test=True):
        ''' evluates the measurements on specified splits (default all).

            whole_dataset: is a lightning data module that includes data for train, val and test.
            If one is not needed None can be sent and None willl be returned.
            
            on_[dataset_subset]: a boolean variable indicating whether to use this split.
            
            
            returns the results as a list [train_results, val_results, test_results] each one a list of [result_of_on_clean, result_of_on_attack].
            if a measure doesn't implement on_clean or on_attack returns None as the result. 
        '''

        train_results = val_results = test_results = None
        if on_train:
            print('Measuring on Train set:')
            train_results = self.evaluate_measures(whole_dataset.train_dataloader(), measures, attacks)
        if on_val:
            val_results = self.evaluate_measures(whole_dataset.val_dataloader(), measures, attacks)
        if on_test:
            print('Measuring on Test set:')
            test_results = self.evaluate_measures(whole_dataset.test_dataloader(), measures, attacks)

        return train_results, val_results, test_results



    def robustly_train(self, trainer:Trainer, num_epochs:int, whole_dataset:pl.LightningDataModule, measures: List[Measure], attacks, 
                   train_measure_frequency=100, val_measure_frequency=100):
        ''' Runs the 'train_single_epoch' for 'num_epochs'.
            trainer implements the abstract class Trainer.
            Evaluates on training/validation set with '[train/val]_measure_frequency' (if epoch_index % measure_frequency == 0)
        '''
        train_loader = whole_dataset.train_dataloader()
        val_loader = whole_dataset.val_dataloader()

        if train_loader is None:
            raise TypeError('Train loader was None. you should have a train loader to use robustly_train().')

        # save measurement results
        train_measurement_results = {}
        val_measurement_results = {}

        #progress bar
        pbar = trange(num_epochs)
        for epoch_index in pbar:
            pbar.set_description(f"training epoch {epoch_index}")

            measure_train_model:bool = epoch_index % train_measure_frequency == 0
            train_single_epoch_measurements = self._train_single_epoch(trainer, train_loader, measures, attacks, measure_model=measure_train_model)
            # save results
            if measure_train_model:
                train_measurement_results[epoch_index] = train_single_epoch_measurements

            # get the validation set results 
            if epoch_index % val_measure_frequency ==0:
                pbar.set_description(f"validating epoch {epoch_index}")
                eval_result = self.evaluate_measures(val_loader, measures, attacks)
                val_measurement_results[epoch_index] = eval_result

        
        return train_measurement_results, val_measurement_results
        

    def _train_single_epoch(self, trainer:Trainer, dataloader: DataLoader, measures: List[Measure], attacks, measure_model:bool=False):
        ''' A single epoch of training loop.
            The model is trained on all the adverserial examples created by the attacks.
            measure_model: if true calculates the measures for all data points. 
            later I can add the ability to train using all attacks (just like now), But measure on a subset of attacks we used to train
        '''
        # I don't merge this with the 'measure_on_batch' since I want to train the model on all atacks
        # and then measure the model on all attacks. 
        # otherwise in 'measure_on_batch' we would measure after each attack when the model is only partially trained.
        pbar = tqdm(enumerate(dataloader), leave=False) 
        for batch_index, data in pbar:
            pbar.set_description(f"batch: {batch_index}")
            
            inputs, labels = data[0].to(self.device), data[1].to(self.device)

            # train on all attacks
            self.model.train()
            for attack in attacks:
                # make adversarial images 
                adv_inputs = attack(inputs, labels) # the model should be already sent to init the attack (according to torchattacks)
                adv_outputs = self.model(adv_inputs)
                adv_predictions = self.predictor(adv_outputs) 
                #train the model
                trainer.train(self.model, labels, adv_inputs, adv_outputs, adv_predictions, attack)
            
            if measure_model:
                # measure the model
                self.model.eval()
                # record the result of clean data 
                outputs = self.model(inputs)
                predicted_labels = self.predictor(outputs)
                self._measure_on_batch(measures, attacks, inputs, labels, outputs, predicted_labels)
        
        
        results = None
        if measure_model:
            # get measurement results for the whole dataset
            results = []
            for m in measures:
                results.append(m.final_result())

        self.model.eval()
        return results


    def evaluate_measures(self, dataloader: DataLoader, measures: List[Measure], attacks):
        ''' Called during test and validation.
            Calculates the results of measures on the data.
            calls each measure's on_clean on each batch. 
            calls each measure's on_attack on each batch and for each attack.
        '''
        if dataloader is None:
            return None
    
        for data in tqdm(dataloader):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            outputs = self.model(inputs)
            predicted_labels = self.predictor(outputs)
            # Batch calculations
            self._measure_on_batch(measures, attacks, inputs, labels, outputs, predicted_labels)            
            
        # get results for the whole dataset
        results = []
        for m in measures:
            results.append(m.final_result())

        return results

    def _measure_on_batch(self, measures: List[Measure], attacks, inputs, labels, outputs, predicted_labels):
        ''' Called on each batch to evaluate the results of these attacks and measures
        '''
        # calculate measures that just need the clean data
        for m in measures:
            m.on_clean_data(self.model, inputs, labels, outputs, predicted_labels)

        # calculate the measures that are based on an attack
        for attack in attacks:
            # make adversarial images 
            adv_inputs = attack(inputs, labels) # the model should be already sent to init the attack (according to torchattacks)
            adv_outputs = self.model(adv_inputs)
            adv_predictions = self.predictor(adv_outputs)
            
            # calculate measurements that need the attack data
            for m in measures:
                m.on_attack_data(self.model, inputs, labels, outputs, predicted_labels, adv_inputs, adv_outputs, adv_predictions, attack)
        
        #finilize the result for each batch
        for m in measures:
            m.batch_end()







    
    def make_non_robust_dataset(train_set, attack_method):
        ''' Make a non-robust train set
        '''

        # very similar to measure_models_robust_accuracy the main difference is that having targets 
        # here is mandatory. 
        
        return NotImplementedError # new dataset 

    def make_robust_dataset(train_set):
        return NotImplementedError







