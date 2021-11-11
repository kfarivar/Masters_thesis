from numpy.core.numeric import NaN
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging as log

def add_normalization_layer(model, model_mean, model_std):
    ''' the attacks library requires the inputs to be in [0,1] but the model can have their own input requirements.
        Adding a normalization layer for the model. 
        We can't use torch.transforms because it supports only non-batch images.
    '''

    class Normalize(nn.Module) :
        def __init__(self, mean, std) :
            super().__init__()
            self.register_buffer('mean', torch.Tensor(mean))
            self.register_buffer('std', torch.Tensor(std))
            
        def forward(self, input):
            # Broadcasting
            mean = self.mean.reshape(1, 3, 1, 1)
            std = self.std.reshape(1, 3, 1, 1)
            return (input - mean) / std

    norm_layer = Normalize(mean=model_mean, std=model_std)

    new_model = nn.Sequential(
        norm_layer,
        model
    )

    return new_model


""" def add_linear_layer(model, pre_process:nn.Module, feature_number, class_number, eval_mode:bool=True, disable_grad:bool=True):
    ''' Adds a linear layer to the end of a network (trained using unsupervise learning) for classification.
        feature_number: number of features the unsupervised network creates
        class_number: number of classes in the classification task 
        pre_process: apply any additional preprocessing on the output of the model like flatten
        eval_mode: fix the drop out and batch normalization layers.
        disable_grad: disable the gradient in the backend model
    '''
    encoder = nn.Sequential(
        model,
        pre_process
    )

    if disable_grad:
        encoder.requires_grad_(requires_grad=False)
    # we have to set eval after Sequential otherwise it will be reset to training.
    if eval_mode:
        encoder.eval()

    linear_layer = nn.Linear(feature_number, class_number)

    return encoder, linear_layer """



class normalize(nn.Module):
    '''make functional.normalize a module so we can use it in Sequential.
    '''
    def __init__(self, dim):
        super(normalize, self).__init__()
        self.normal = F.normalize
        self.dim = dim
        
    def forward(self, x):
        x = self.normal(x, dim=self.dim)
        return x


def remove_operation_count_from_dict(state_dict):
    '''remove operators count in state dict. 
       used for the barlow twins model since they use thop library !
       if a key ends in total_ops or total_params it is removed.'''

    import re
    to_remove= []
    for key in state_dict.keys():
        # if it ends in total_ops or total_params remove it !
        if re.search(r"total_ops$|total_params$", key):
            to_remove.append(key)

    for key in to_remove:
        state_dict.pop(key) # if you don't want it to throw error use d.pop(k, None)

    return state_dict
    



import pandas as pd

def save_measurements_to_csv(results, measurements, file_name=None, on_train=True, on_val=True, on_test=True, save=True, epoch=-1):
    ''' Saves(if save is set) the results of the benchmark from the dictionaries to a CSV file.
        results should be the triplet of (train_results, val_results ,test_results)
        returns a pandas df.
        if a epoch >=0 is given adds it as a column

        
    '''
    
    df = pd.DataFrame(columns = ['mode', 'measure_name',  'attack_name', 'descriptor1', 'descriptor2', 'value'])

    def parse_results(mode, df):
        mode_index = {'Train':0, 'Validation':1, 'Test':2}
        for measure_idx, measure in enumerate(measurements):
            # some measures return a single result 
            if isinstance(results[mode_index[mode]][measure_idx], dict):
                for attack, descriptors in results[mode_index[mode]][measure_idx].items():
                    # some measurements dont have descriptors !
                    if isinstance(descriptors, dict):
                        for i, (descriptor, value) in enumerate(descriptors.items()):
                            row = {}
                            row['mode'] = mode
                            row['measure_name'] = type(measure).__name__
                            row['attack_name'] = attack
                            row[f'descriptor{i+1}'] = descriptor
                            row['value'] = value
                            df = df.append(row, ignore_index=True)
                    else:
                        df = df.append({'mode':mode, 'measure_name':type(measure).__name__, 'attack_name':attack, 'value':descriptors}, ignore_index=True)
            else:
                df = df.append({'mode':mode, 'measure_name':type(measure).__name__, 'value':results[mode_index[mode]][measure_idx]}, ignore_index=True)
        
        return df

    if on_train and results[0] is not None:
        df = parse_results('Train', df)

    if on_val and results[1] is not None:
        df = parse_results('Validation', df)
        
    if on_test and results[2] is not None:
        df = parse_results('Test', df)

    if save:
        df.to_csv(file_name, index=False)

    if epoch >=0:
        df['epoch'] = epoch
        
    return df


def save_training_results_to_csv(results, measurements, file_name):
    train_val_result, test_result = results

    train_measurement_results, val_measurement_results = train_val_result

    train_dfs = []
    for epoch,v in train_measurement_results.items():
        # add epoch as a column
        df = save_measurements_to_csv((v,None, None), measurements, on_val=False, on_test=False, save=False, epoch= epoch)
        train_dfs.append(df)
    
    val_dfs = []
    for epoch,v in val_measurement_results.items():
        # add epoch as a column
        df = save_measurements_to_csv((None,v, None), measurements, on_train=False, on_test=False, save=False, epoch= epoch)
        val_dfs.append(df)

    all_dfs = train_dfs + val_dfs
    if test_result is not None:
        test_df = save_measurements_to_csv((None,None, test_result), measurements, on_val=False, on_train=False, save=False, epoch= 0)
        all_dfs.append(test_df)
    
    
    all_results = pd.concat(all_dfs, ignore_index=True)
    #reorder columns
    columns = ['epoch','mode', 'measure_name',  'attack_name', 'descriptor1', 'descriptor2', 'descriptor3', 'descriptor4',  'value']
    all_results = all_results[columns]

    # save as csv
    all_results.to_csv(file_name, index=False)

    return all_results


def print_measurement_results(results, measurements, on_train=True, on_val=True, on_test=True, set_log_stream=False):
    ''' Prints the results of the benchmark properly
    '''

    if set_log_stream:
        log.basicConfig(
            level=log.DEBUG,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                log.StreamHandler()
            ]
        )


    if on_train and results[0] is not None:
        log.info('Train set results:')
        for idx, m in enumerate(measurements):
            log.info(f'The {type(m).__name__} results:')
            log.info(results[0][idx])
        log.info('-'*20)
    if on_val and results[1] is not None:
        log.info('Validation set results:')
        for idx, m in enumerate(measurements):
            log.info(f'The {type(m).__name__} results:')
            log.info(results[1][idx])
        log.info('-'*20)
    if on_test and results[2] is not None:
        log.info('Test set results:')
        for idx, m in enumerate(measurements):
            log.info(f'The {type(m).__name__} results:')
            log.info(results[2][idx])
        log.info('-'*20)


def print_train_test_val_result(results, measurements):
    train_val_result, test_result = results

    train_measurement_results, val_measurement_results = train_val_result

    for epoch,v in train_measurement_results.items():
        log.info(f'epoch {epoch} train result:')
        print_measurement_results((v,None, None), measurements, on_val=False, on_test=False)
    
    log.info('*'*20)
    
    for epoch,v in val_measurement_results.items():
        log.info(f'epoch {epoch} validation result:')
        print_measurement_results((None,v, None), measurements, on_train=False, on_test=False)

    log.info('*'*20)

    # print the test result
    if test_result is not None:
        print_measurement_results((None,None, test_result), measurements, on_val=False, on_train=False)
        log.info('*'*20)

