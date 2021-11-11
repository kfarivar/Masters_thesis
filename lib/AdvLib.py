import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging as log
import pytorch_lightning as pl
from typing import List
from tqdm import tqdm
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

from .Measurements import Measure, Dataset_measure
from .Trainer import Trainer
from .utils import save_measurements_to_csv
import pandas as pd



class Adversarisal_bench:
    '''the main class that the user is going to interact with '''

    def __init__(self, model:nn.Module, predictor, tb_board_save_dir=None, untrained_state_dict=None, device = 'cuda:0'):
        ''' model: Takes pretrained model. The model should have the forward method implemented
            predictor: the function that transforms the output of the network into labels. takes inputs in batches.
            untrained_state_dict: to start the training from scratch.
            tb_board_save_dir: save path to save tensorboard logs .
        '''
        self.model = model
        self.predictor = predictor
        self.untrained_state_dict = untrained_state_dict
        self.device = device
        # initialize
        self.model.eval().to(self.device)
        self.model.requires_grad_(False)
        log.info(f'device is: {self.device}')

        self.ts_writer = None
        if tb_board_save_dir is not None:
            self.ts_writer = SummaryWriter(tb_board_save_dir)


        
    def train_val_test(self, trainer:Trainer, num_epochs:int, whole_dataset:pl.LightningDataModule,measures:List[Measure], 
                        attacks, save_path, train_measure_frequency=100, val_measure_frequency=100, run_test=True, reset_model=False):
        ''' Uses 'robustly_train' function to train and validate and 'evaluate_measures' to test.
            Warning: the network sent to the benchmark will change after calling this function. save using new_model=copy.deepcopy(old_model)
            The model sent to the benchmark will be modifed to get the robust model.
            reset_model(bool): where to start the training 
                True: start from the pretrained model sent to AdvLib 
                False: set all the weights to self.untrained_state_dict then train  
            Evaluates on training/validation set with '[train/val]_measure_frequency' (if epoch_index % measure_frequency == 0)
            saves the model at every evaluation at save_path
        '''
        if reset_model:
            self.model.load_state_dict(self.untrained_state_dict)

        print('Training:')
        train_val_result = self.robustly_train(trainer, num_epochs, whole_dataset, measures, attacks, save_path, 
                                                train_measure_frequency, val_measure_frequency)

        test_result = None
        if run_test:
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
            print('Measuring on Validation set:')
            val_results = self.evaluate_measures(whole_dataset.val_dataloader(), measures, attacks)
        if on_test:
            print('Measuring on Test set:')
            test_results = self.evaluate_measures(whole_dataset.test_dataloader(), measures, attacks)

        return train_results, val_results, test_results


    def robustly_train(self, trainer:Trainer, num_epochs:int, whole_dataset:pl.LightningDataModule, measures: List[Measure], attacks, save_path,
                   train_measure_frequency=100, val_measure_frequency=100):
        ''' Runs the 'train_single_epoch' for 'num_epochs'.
            trainer implements the abstract class Trainer.
            Evaluates on training/validation set with '[train/val]_measure_frequency' (if epoch_index % measure_frequency == 0)
            saves only the epochs that are evaluated.
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
            train_single_epoch_measurements = self._train_single_epoch(trainer, train_loader, measures, attacks, epoch_index, measure_model=measure_train_model)
            # save results
            if measure_train_model:
                train_measurement_results[epoch_index] = train_single_epoch_measurements

            # get the validation set results 
            if epoch_index % val_measure_frequency ==0:
                pbar.set_description(f"validating epoch {epoch_index}")
                eval_result = self.evaluate_measures(val_loader, measures, attacks)
                val_measurement_results[epoch_index] = eval_result
                # log results 
                self.log_evaluate_results(eval_result, measures, epoch_index, 'Validation')
                # save the model
                torch.save(self.model.state_dict(), save_path+f'/epoch_{epoch_index}.pt')


        
        return train_measurement_results, val_measurement_results
        

    def _train_single_epoch(self, trainer:Trainer, dataloader: DataLoader, measures: List[Measure], attacks, epoch, measure_model:bool=False):
        ''' A single epoch of training loop.
            The model is trained on all the adverserial examples created by the attacks.
            measure_model: if true calculates the measures for all data points. 
            later I can add the ability to train using all attacks (just like now), But measure on a subset of attacks we used to train
        '''
        # I don't merge this with the 'measure_on_batch' since I want to train the model on all attacks
        # and then measure the model on all attacks. 
        # otherwise in 'measure_on_batch' we would measure after each attack when the model is only partially trained.

        
        
        pbar = tqdm(enumerate(dataloader), leave=False) 
        for batch_index, data in pbar:
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            # an index to identify each individual data point
            indexes = data[2]
            # train on all attacks
            self.model.train()
            for attack in attacks:
                # save losses of different attacks
                loss_list = []
                # make adversarial images set model flags before/after to make sure
                self.model.requires_grad_(False)
                self.model.eval()
                adv_inputs = attack(inputs, labels).to(self.device) # the model should be already sent to init the attack (according to torchattacks)
                self.model.requires_grad_(True)
                self.model.train()
                # forward pass
                adv_outputs = self.model(adv_inputs)
                adv_predictions = self.predictor(adv_outputs)
                #train the model
                loss = trainer.train(self.model, labels, adv_inputs, adv_outputs, adv_predictions, attack)
                loss_list.append(loss.item())

            # show results
            pbar.set_description(f"Epoch [{epoch}], Step [{batch_index}], Avg Loss of attacks: {np.mean(loss_list)/labels.size(0)}")
            if self.ts_writer is not None:
                self.ts_writer.add_scalar('Train/loss_per_step', np.mean(loss_list)/labels.size(0), epoch * len(dataloader) + batch_index)
            

        # Keep the model fixed and do measurements on the train set 
        results = None  
        if measure_model:
            # measure the model
            results = self.evaluate_measures(dataloader, measures, attacks)
            # log results 
            self.log_evaluate_results(results, measures, epoch, 'Train')

        return results

    
    def log_evaluate_results(self, results, measures, epoch, mode):
        '''log to tensor board'''
        # get results df and log them
        if self.ts_writer is not None:
            results_df = save_measurements_to_csv((results, None, None), measures, save=False)
            for index, row in results_df.iterrows():
                result = row['value']
                row = row.dropna().drop(labels=['mode', 'value'])
                # concatenate the rest of discriptors as label
                label = ''
                for key, value in row.items():
                    label += str(value) + '_'
                label = label[:-1]
                # log to tensorboard
                self.ts_writer.add_scalar(f'{mode}/{label}', result, epoch)


    def evaluate_measures(self, dataloader: DataLoader, measures: List[Measure], attacks):
        ''' Called during test and validation.
            Calculates the results of measures on the data.
            calls each measure's on_clean on each batch. 
            calls each measure's on_attack on each batch and for each attack.
        '''
        if dataloader is None:
            return None

        self.model.requires_grad_(False)
        self.model.eval()

        for m in measures:
            m.before_evaluation(self.model)

        for data in tqdm(dataloader):
            with torch.no_grad():
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                # an index to identify each individual data point
                indexes = data[2]
                outputs = self.model(inputs)
                predicted_labels = self.predictor(outputs)
            # Batch calculations
            self._measure_on_batch(measures, attacks, inputs, labels, outputs, predicted_labels, indexes)            
            
        # get results for the whole dataset
        results = []
        for m in measures:
            results.append(m.final_result())

        return results


    def _measure_on_batch(self, measures: List[Measure], attacks, inputs, labels, outputs, predicted_labels, indexes):
        ''' Called on each batch to evaluate the results of these attacks and measures
        '''
        # calculate measures that just need the clean data
        with torch.no_grad():
            for m in measures:
                m.on_clean_data(self.model, inputs, labels, outputs, predicted_labels, indexes)

        # calculate the measures that are based on an attack
        for attack in attacks:
            # make adversarial images (here we need the grad for input gradients !)
            adv_inputs = attack(inputs, labels) # the model should be already sent to init the attack (according to torchattacks)

            with torch.no_grad():
                adv_outputs = self.model(adv_inputs)
                adv_predictions = self.predictor(adv_outputs)
                # calculate measurements that need the attack data
                for m in measures:
                    m.on_attack_data(self.model, inputs, labels, outputs, predicted_labels, 
                                    adv_inputs, adv_outputs, adv_predictions, attack, indexes)
        
        #finilize the result for each batch
        for m in measures:
            m.batch_end()



    # needs debuging
    def measure_on_whole_dataset(self, measures:List[Dataset_measure], whole_dataset:pl.LightningDataModule, split='test'):
        ''' This is for measures that require access to the whole dataset 
            and are harder to implement batch wise. (e.g methods that include clustering or Knn).

            split: either 'test', 'train' or 'val'  
        '''
        if split =='train':
            self._measure_on_dataset_split(measures, whole_dataset.train_dataloader())

        elif split == 'val':
             self._measure_on_dataset_split(measures, whole_dataset.val_dataloader())

        else:
            self._measure_on_dataset_split(measures, whole_dataset.test_dataloader())

    # needs debuging
    def _measure_on_dataset_split(self, measures, dataloader: DataLoader):
        ''' The method that actually calculates the datatset specific measures.
        '''
        if dataloader is None:
            return None

        self.model.eval()
    
        print("gathering data points ...")
        for data in dataloader:
            inputs = torch.cat((inputs, data[0].to(self.device)), dim=0) 
            labels = torch.cat((inputs, data[1].to(self.device)), dim=0) 
        print(inputs.shape)
        print(labels.shape)

        print("measuring on dataset ...")
        results = []
        for m in measures:
            r = m.on_data(inputs, labels)
            results.append(r)
            

        # get results for the whole dataset
        return results















