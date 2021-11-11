import numpy as np
import torch
import torch.nn as nn
import attr
from abc import ABC, abstractmethod
from copy import deepcopy

from art.metrics import clever_u
from art.estimators.classification import PyTorchClassifier

class Measure(ABC):
    ''' A specific measurement to be applied on a batch of data in measure method of AdvLib.
    '''

    def before_evaluation(*args, **kwargs):
        ''' called right before the loop on dataset to eavluate the model. 
        mostly useful to register hooks to examine mid layer activations for the rest of the code.
        '''
        pass
    
    @abstractmethod
    def on_clean_data():
        '''called on the result of a batch of data. clled once per batch. counting number of points should happen here.
        '''
        pass

    @abstractmethod
    def on_attack_data():
        ''' called on the result of an attack on a batch of data. called the same number of times as number of attacks.
            Any attack related statistics should be calculated here preferrebly saved in a dictionary keyed by attack name.
        '''
        pass

    
    def batch_end(*args, **kwargs):
        '''called after calcuating each single batch.
        '''
        return None

    @abstractmethod
    def final_result():
        '''called after the whole dataset is processed.
        '''
        pass


class Feature_diff(Measure):
    ''' Calculate the Euclidean norm of the difference of a layer before and after attack for each sample.
        return the average over the whole dataset. 
    '''

    def __init__(self, layer):
        ''' layer is the name of the layer we want to calculate the difference for.
            The name of each layer can be found using:
                for name, layer in model.named_modules():
            layer can be set to 'output' to consider the logits of the model. 
        '''
        self.layer = layer
        # total num of data points
        self.total_count = 0
        # sum of the eculidean distances of differences of layer for each attack
        self.dist_sums = {}
        # save the output of layer after each forward
        self.activation = None
        # activations for clean data
        self.clean_activations = None
        # save handle to hook to free it at the end
        self.hook_handle = None 

    def before_evaluation(self, model):
        # the normal logits outputs don't need hooks
        if self.layer != 'output':
            # hook to be sent to the layer
            def save_activations(model, input, output):
                self.activation = output.detach()
            #send the hook to the target layer
            layer_exists = False
            for name, layer_module in model.named_modules():
                if name == self.layer:
                    self.hook_handle = layer_module.register_forward_hook(save_activations)
                    layer_exists = True
        if not layer_exists:
            raise ValueError('The layer name sent to Feature_diff is incorrect !!')
                    


    def on_clean_data(self, model, inputs, labels, outputs, predicted_labels, indexes):
        self.total_count += outputs.size(0)
        # save clean activations
        self.clean_activations = deepcopy(self.activation)


    def on_attack_data(self, model, inputs, labels, outputs, predicted_labels, adv_inputs, adv_output, adv_predictions, attack, indexes):
        ''' Calculate the sum of norms of a batch '''

        if self.layer == 'output':
            diff = outputs - adv_output
        else: # self.activation should be activations for the attack sample
            diff = self.clean_activations - self.activation  

        # flatten the diff starting from the second dimesnion (in case it is a conv layer output)
        diff = torch.flatten(diff, start_dim=1)
        norms = torch.norm(diff, dim=1, p=2)

        attack_name = attack.get_name()
        self.dist_sums[attack_name] = self.dist_sums.get(attack_name, 0) + torch.sum(norms)
    
    
    def final_result(self):
        result = {}
        for attack_name, norm_sum in self.dist_sums.items():
            result[attack_name] = (norm_sum / self.total_count).item()
        
        # reset the variabel
        self.total_count = 0
        self.dist_sums = {}
        self.activation = None
        self.clean_activations = None

        #free the hook to avoid adding many hooks durin training
        if self.layer != 'output':
            self.hook_handle.remove()
        self.hook_handle = None

        return result


        



@attr.s
class Normal_accuracy(Measure):
    ''' Measure a specific models non-robust / normal accuracy.
    '''
    # total number of data points
    _total = attr.ib(default=0)
    # number of corrcet predictions
    _correct = attr.ib(default=0) 

    def on_clean_data(self, model, inputs, labels, outputs, predicted_labels, indexes):
        self._total += labels.size(0)
        self._correct += (predicted_labels == labels).sum().item()


    def on_attack_data(self,*args):
        return None

        
    def final_result(self):
        accuracy = self._correct/self._total

        ############################
        # IMPORTANT
        #reset the values
        self._total = 0
        self._correct = 0

        return accuracy *100
        
class Loss_measure(Measure):
    ''' Caclulate the specified loss'''
    def __init__(self, loss_function):
        '''loss_function: like a pytorch loss function should accept logits and label in batch form'''
        super().__init__()

        self.loss_func = loss_function
        # total num of data points
        self.total_count = 0

        # sum of losses for clean data and different attacks
        self.sum_losses = {'Clean':0.0}
        
    def on_clean_data(self, model, inputs, labels, outputs, predicted_labels, indexes):
        self.total_count += labels.size(0)
        clean_loss = self.loss_func(outputs, labels)
        self.sum_losses['Clean'] += clean_loss

    def on_attack_data(self, model, inputs, labels, outputs, predicted_labels, adv_inputs, adv_output, adv_predictions, attack, indexes):
        '''calculate loss separately for each attack.'''
        attack_loss = self.loss_func(adv_output, labels)
        attack_name = attack.get_name()
        self.sum_losses[attack_name] = self.sum_losses.get(attack_name, 0.0) + attack_loss

    def final_result(self):
        results = {}
        for attack_name, sum_loss in self.sum_losses.items():
            results[attack_name] = sum_loss / self.total_count
        
        # reset the counters
        self.total_count = 0
        self.sum_losses = {'Clean':0.0}

        return results

from torch import linalg as LA
class Check_perturbation(Measure):
    '''check if the adv example is in the esilon neighbourhood of the sample'''
    def __init__(self, eps, norm):
        '''
        eps: the purturbation
        norm: either 'linf' or 'l2'
        '''
        super().__init__()
        self.eps = eps
        # total num of data points
        self.norm = norm

        # are all purturbations in bound
        self.all_in_bound = {'Clean':0.0}
    
    def on_clean_data(self, *args):
        return None
    
    def on_attack_data(self, model, inputs, labels, outputs, predicted_labels, adv_inputs, adv_output, adv_predictions, attack, indexes):
        # get the 2D matrix of differences dim 0 is batch
        diff = torch.flatten(inputs - adv_inputs, start_dim=1)
        if self.norm == 'l2':
            norm_of_diff = LA.vector_norm(diff, ord=2, dim=1)
        elif self.norm == 'linf':
            norm_of_diff = LA.vector_norm(diff, ord=np.inf, dim=1)

        # check in bound
        mask = norm_of_diff > self.eps

        if mask.any():
            print('the perturbation for images at indexes below are out of bound:')
            print(indexes[mask])
            print('Amount of perturbations:')
            print(norm_of_diff[mask])
            raise ValueError('Perturbation out of bound.')

        return None


    def final_result(self):
        return None






@attr.s
class Robust_accuracy(Measure):
    ''' Measures the accuracy of the model with respect to a certain attack 
    '''
    # total number of data points
    _total = attr.ib(default=0)
    # total number of data points that are correctly predicted Before applying the attack
    _total_correct = attr.ib(default=0)


    # number of data points that were correctly predicted after attack (regadrless of, if they were initially predicted correctly or not)
    # the keys are the attack names attack.get_name()
    _correct_after_attack = attr.ib(factory=dict)
    
    # number of data points that were corrcetly predicted both before and after applying the attack
    _correct_before_and_after_attack = attr.ib(factory=dict) 

    def on_clean_data(self, model, inputs, labels, outputs, predicted_labels, indexes):
        self._total += labels.size(0)
        self._total_correct += (predicted_labels == labels).sum().item()



    def on_attack_data(self, model, inputs, labels, outputs, predicted_labels, adv_inputs, adv_output, adv_predictions, attack, indexes):
        ''' Takes a batch and the adverserial version of the batch and returns what precent of the previously correct are correct after the
            attack and what percent of total are now correct.
        '''
        # separate sampels that are correctly classified
        mask_correct = (labels == predicted_labels)
        correctly_predicted_labels = labels[mask_correct]
        corresponding_adv_predictions = adv_predictions[mask_correct]
        attack_name = attack.get_name()
        # count the total correct after this attack (add the key to dictionary if not there)
        self._correct_after_attack[attack_name] = self._correct_after_attack.get(attack_name, 0) + (labels == adv_predictions).sum().item()
        # count the total that were correct before and after the attack
        self._correct_before_and_after_attack[attack_name] = self._correct_before_and_after_attack.get(attack_name,0) + \
                                                            (correctly_predicted_labels == corresponding_adv_predictions).sum().item()

        
    def final_result(self):
        results = {}
        for attack_name, correct_after_attack in self._correct_after_attack.items():
            correct_before_and_after_attack = self._correct_before_and_after_attack[attack_name]
            #initialise a sub dict for this attack's result
            results[attack_name] = {}
            # return the two stats
            results[attack_name]['Total_accuracy'] = correct_after_attack / self._total * 100
            results[attack_name]['Correct_accuracy'] = correct_before_and_after_attack / self._total_correct * 100

        # reset the counters
        self._total = 0
        self._total_correct = 0
        self._correct_after_attack = {}
        self._correct_before_and_after_attack = {}

        return results


@attr.s
class Subset_saliency(Measure):
    pass



# needs debuging
@attr.s
class Clever_score(Measure):
    ''' Note, the implemnetation of the adversarial-robustness-toolbox is inefficient and incomplete !  
    Clever is an estimate of the minimum amount of purturbation required to create an adverserial example for a specific data point.
    This measure returns the average score for a subset of the data set. 
    see paper: https://arxiv.org/abs/1801.10578
    The adversarial-robustness-toolbox implementation is used. (https://github.com/Trusted-AI/adversarial-robustness-toolbox)
    
    The number of classes should match the second dimension of outputs.  
    '''
    #a set of indexes for the subset of points to calculate the score on
    # array-like (e.g. list)
    data_point_indexes = attr.ib()
    #The loss function used for training (e.g. for classification torch.nn.CrossEntropyLoss())
    loss = attr.ib()
    # refer to library docs and paper
    nb_batches:int=attr.ib()
    batch_size:int=attr.ib()
    # radius similar to the epsilon in an attack
    radius:float = attr.ib()
    # either 1,2,np.inf
    norm = attr.ib()
    c_init = attr.ib()
    pool_factor = attr.ib()
    verbose = attr.ib()


    #(internal) dict of the clever scores. index is key clever score is value.
    _clever_scores = attr.ib(default=dict()) 

    def on_clean_data(self, model, inputs, labels, outputs, predicted_labels, indexes):
        # load the model into the librray format
        classifier = PyTorchClassifier(
                                    model=model,
                                    loss=self.loss,
                                    input_shape=list(inputs.shape[1:]),
                                    nb_classes=outputs.shape[1],
                                ) 

        # get indexes we are interested in.
        # access_indexes is the index we can use to get the desired input
        data_indexes, _, access_indexes = np.intersect1d(self.data_point_indexes, indexes)
        # this for loop can be parallelized in cpu
        for data_idx, access_idx in zip(data_indexes,access_indexes):
            # data points can be duplicates avoid double calc
            if data_idx not in self._clever_scores:
                # get the access index
                data_point = inputs[access_idx].numpy() 

                clever_score = clever_u(classifier=classifier , x=data_point, nb_batches=self.nb_batches, batch_size=self.batch_size, 
                                radius=self.radius, norm=self.norm, c_init=self.c_init, pool_factor=self.pool_factor, verbose=self.verbose)

                self._clever_scores[data_idx] = clever_score



    def on_attack_data(self,*args):
        return None

    def batch_end(self):
        return None
        
    def final_result(self):
        avg_clever_score = np.mean(list(self._clever_scores.values()))
        ############################
        # IMPORTANT
        #reset the values
        self._clever_scores = dict()

        return avg_clever_score
    



class Dataset_measure(ABC):
    ''' for measures specific to the dataset. 
        These usually require the whole dataset to calculate things like knn or clustering.
    '''
    @abstractmethod
    def on_data():
        pass

        





        
