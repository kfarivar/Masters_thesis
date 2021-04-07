import torch
import torch.nn as nn
import torch.tensor
import attr

from abc import ABC, abstractmethod

class Measure(ABC):
    ''' A specific measurement to be applied on a batch of data in measure method of AdvLib.
    '''
    
    @abstractmethod
    def on_clean_data():
        '''called on the result of a batch of data.
        '''
        pass

    @abstractmethod
    def on_attack_data():
        ''' called on the result of an attack on a batch of data.
        '''
        pass

    @abstractmethod
    def batch_end():
        '''called after calcuating each single batch.
        '''
        pass

    @abstractmethod
    def final_result():
        '''called after the whole dataset is preocessed.
        '''
        pass



@attr.s
class Normal_accuracy(Measure):
    ''' Measure a specific models non-robust / normal accuracy.
    '''
    # total number of data points
    _total = attr.ib(default=0)
    # number of corrcet predictions
    _correct = attr.ib(default=0) 

    def on_clean_data(self, model, inputs, labels, outputs, predicted_labels):
        self._total += labels.size(0)
        self._correct += (predicted_labels == labels).sum().item()


    def on_attack_data(self,*args):
        return None

    def batch_end(self):
        return None
        
    def final_result(self):
        accuracy = self._correct/self._total

        ############################
        # IMPORTANT
        #reset the values
        self._total = 0
        self._correct = 0

        return accuracy
        


@attr.s
class Robust_accuracy(Measure):
    ''' Measures the accuracy of the model with respect to a certain attack 
    '''
    # total number of data points
    _total = attr.ib(default=0)
    # total number of data points that are correctly predicted Before applying the attack
    _total_correct = attr.ib(default=0)


    # number of data points that were correctly predicted after attack (regadrless of, if they were initially predicted correctly or not)
    # the keys are the attack names type(attack).__name__
    _correct_after_attack = attr.ib(factory=dict)
    
    # number of data points that were corrcetly predicted both before and after applying the attack
    _correct_before_and_after_attack = attr.ib(factory=dict) 

    def on_clean_data(self, model, inputs, labels, outputs, predicted_labels):
        self._total += labels.size(0)
        self._total_correct += (predicted_labels == labels).sum().item()



    def on_attack_data(self, model, inputs, labels, outputs, predicted_labels, adv_inputs, adv_output, adv_predictions, attack):
        ''' Takes a batch and the adverserial version of the batch and returns what precent of the previously correct are correct after the
            attack and what percent of total are now correct.
        '''
        # separate sampels that are correctly classified
        mask_correct = (labels == predicted_labels)
        correctly_predicted_labels = labels[mask_correct]
        corresponding_adv_predictions = adv_predictions[mask_correct]
        attack_name = type(attack).__name__
        # count the total correct after this attack
        self._correct_after_attack[attack_name] = self._correct_after_attack.get(attack_name, 0) + (labels == adv_predictions).sum().item()
        # count the total that were correct before and after the attack
        self._correct_before_and_after_attack[attack_name] = self._correct_before_and_after_attack.get(attack_name,0) + \
                                                            (correctly_predicted_labels == corresponding_adv_predictions).sum().item()

    
    def batch_end(self):
        return None
        
    def final_result(self):
        results = {}
        for attack_name, correct_after_attack in self._correct_after_attack.items():
            correct_before_and_after_attack = self._correct_before_and_after_attack[attack_name]
            #initialise a sub dict for this attack's result
            results[attack_name] = {}
            # return the two stats
            results[attack_name]['percentage_correct_after_attack'] = correct_after_attack / self._total * 100
            results[attack_name]['percentage_of_correct_that_remain_correct_after_attack'] = correct_before_and_after_attack / self._total_correct * 100

        # reset the counters
        self._total = 0
        self._total_correct = 0
        self._correct_after_attack = {}
        self._correct_before_and_after_attack = {}

        return results




if __name__ == '__main__':
    # tests
    ra = Robust_accuracy()

    labels = torch.tensor([1,0,1,1,1])
    predicted_labels = torch.tensor([1,0,1,1,0]) # 80% accuracy
    adv_predictions = torch.tensor([0,1,1,1,0]) # 20% both
    import torchattacks
    from simple_model import Net
    attack = torchattacks.PGD(Net(), eps=8/255, alpha=2/255, steps=4) 

    ra.on_attack_data(None, None, labels, None, predicted_labels, None, None, adv_predictions, attack)

    print(ra)
        

# TODO
""" class Concentration_measure(Measure):
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
        return None """

        
