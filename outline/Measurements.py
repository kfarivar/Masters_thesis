import torch
import torch.nn as nn
import torch.tensor
from torch.utils.data import Dataset, DataLoader

from abc import ABC, abstractmethod

class measure(ABC):

    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def on_clean_data(self, whole_dataset:(DataLoader,DataLoader)):
        '''whole_dataset: a tuple of trainset testset.
        '''
        pass

    @abstradctmethod
    def on_attack_data(self, whole_dataset:(DataLoader,DataLoader)):
        pass


class Dataset_measure(measure):
    '''Abstract class for measurements concenrning the dataset
    '''
    pass

class Concentration_measure(Dataset_measure):
    def __init__(self):
        super().__init__()
    
    def on_clean_data(self, whole_dataset:(DataLoader,DataLoader)):
        # input: DataLoader
        # join the test and train
        # 1. calculate knn (preliminary.py)
        # 2. run the proposed algorithm that finds a robust error region (main_infinity.py)
        # return a number
        return NotImplementedError

    def on_attack_data(self, whole_dataset:(DataLoader,DataLoader)):
        return None



class Model_measure(measure):
    ''' Abstarct class for measurements that try to measure the accuracy of a model
    ''' 
    @abstractmethod
    def __init__(self,  model:nn.Module, predictor):
        ''' predictor: is a the function that takes the output of the model and produces labels (e.g torch.max in case of softmax layer).
        '''
        super().__init__()
        self.model= model
        self.predictor = predictor

    @abstractmethod
    def on_clean_data():
        pass

    @abstradctmethod
    def on_attack_data():
        pass


class Normal_accuracy(Model_measure):
    ''' Measure a specific models non-robust / normal accuracy
    '''

    def __init__(self, model:nn.Module, predictor):
        super().__init__(model, predictor)
        

    def on_clean_data(self, whole_dataset:(DataLoader,DataLoader)):
        correct = 0
        total = 0
        
        for data in whole_dataset[1]:
            images, labels = data
            outputs = self.model(images)
            predicted_labels = self.predictor(outputs)
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()

        return 100 * correct / total

    def on_attack_data(self, whole_dataset:(DataLoader,DataLoader)):
        return None



class Robust_accuracy(Model_measure):
    ''' Measures the accuracy of the model with respect to a certain attcak 
    '''
    def __init__(self, model:nn.Module, predictor, attacks:list[Attack]):
        '''attacks: a list of attacks used to check robust accuracy of each sample
        '''
        super().__init__(model, predictor)
        self.attacks = attacks
        

    def on_clean_data(self, whole_dataset:(DataLoader,DataLoader)):
        return None

    def on_attack_data(self, whole_dataset:(DataLoader,DataLoader)):
        '''test_set: is a torch.utils.data.Dataloader. 
           usually implemented using torch.utils.data.Dataset (https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
           The advantage is we can use Dataparallel to speed up the process up (https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html?highlight=dataparallel)

           returns: Tuple (array of precentage of attacks (per sample) failed for each attack, the adverserial example)
        '''
        adverserial_examples = []
        #save the number of corrects for each attack (number of failed attacks)
        failed_attacks = np.zeros(len(self.attacks))
        # assuming batch size is 1 (needs to be genralized)
        for batch in whole_dataset[1]:
            data, label = batch

            # predict
            output = self.predictor(self.model(data))
            #forget about the examples that are already wrong
            if output != label:
                continue

            # iterate through all attacks and save results
            for attack_idx, attack_method in enumerate(self.attacks):
                # result will be the distorted sample and the new class
                adv_images = attack_method(data, label)
                attack_name = type(attack_method).__name__
                # check the result
                if adv_images is None:
                    # print the failed attack
                    print(attack_name + "attack failed")
                    failed_attacks[attack_idx]+=1
                else :
                    # add the attack result to a list
                    adverserial_examples.append((attack_name, data, label, adv_images))

        # return percentage of attacks that didn't succeed and the samples that did fool
        return (failed_attacks/test_set.size(), adverserial_examples)  

        


        
