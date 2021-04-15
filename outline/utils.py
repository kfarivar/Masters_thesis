import torch
import torch.nn as nn

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





def print_dict_pretty(results, measurements, on_train=True, on_val=True, on_test=True):
    ''' Prints the results of the benchmark properly
    '''
    if on_train and results[0] is not None:
        print('Train set results:')
        for idx, m in enumerate(measurements):
            print(f'The {type(m).__name__} results:')
            print(results[0][idx])
        print()
    if on_val and results[1] is not None:
        print('Validation set results:')
        for idx, m in enumerate(measurements):
            print(f'The {type(m).__name__} results:')
            print(results[1][idx])
        print()
    if on_test and results[2] is not None:
        print('Test set results:')
        for idx, m in enumerate(measurements):
            print(f'The {type(m).__name__} results:')
            print(results[2][idx])