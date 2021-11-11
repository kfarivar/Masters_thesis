import torch
import torch.nn as nn

from abc import ABC, abstractmethod



class Attack(ABC):
    '''abstract class for different types of attacks for adding a call function and an identifying name'''

    @abstractmethod
    def __init__(self, adversary):
        pass

    @abstractmethod
    def __call__(self, sample, label):
        '''Should return a bacth of adversarial data'''
        pass
    
    @abstractmethod
    def get_name():
        '''returns the attack name this is used as an identifier when accumulating the results and should be unique !'''
        pass

class ART_Wrapper(Attack):
    '''wrapper for attacks in Adversarial Robustness Tollbox (IBM)
    This librray has major issues in the sense that they don't support gpu devices !!
    '''
    def __init__(self, adversary, name):
        self.adversary = adversary
        self._name = name

    def __call__(self, sample, label):
        return self.adversary.generate(sample.cpu().numpy(), label.cpu().numpy())

    def get_name(self):
        return self._name

    

class Torchattacks_Wrapper(Attack):
    ''' wrapper for torch attachs to add the get_name method '''
    def __init__(self, adversary, name):
        self.adversary = adversary
        self._name = name

    def __call__(self, sample, label):
        return self.adversary(sample, label)

    def get_name(self):
        return self._name
        

class AutoAttack_Wrapper(Attack):
    ''' wrapper for auto-attacks to add the call function
        still need to debug putting the model in or out of eval/train for robust training !!!!
    '''

    def __init__(self, adversary, name):
        ''' The adversary as defined by AutoAttack(forward_pass, norm='norm', eps=epsilon, version='version') 
            use adversary.attacks_to_run = ['apgd-ce'] to select subsets !
        '''
        self.adversary = adversary
        self._name = name

    def __call__(self, samples, labels):
        x_adv = self.adversary.run_standard_evaluation(samples, labels, bs=labels.shape[0])
        return x_adv 

    def get_name(self):
        return self._name



        




