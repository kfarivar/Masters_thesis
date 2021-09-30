import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from torchinfo import summary




from models.simclr_module import SimCLR

device = 'cuda:1'




weight_path = './model_checkpoints/simCLR_unsupervised/epoch=285-step=50335.ckpt'
""" state_dict = torch.load(weight_path)['state_dict'] # get 'state_dict' from lightining checkpoint
weight_key = "non_linear_evaluator.block_forward.2.weight"
bias_key = "non_linear_evaluator.block_forward.2.bias"
# load the linear layer weights from unsupervise training
weight = state_dict[weight_key].to(device)
bias = state_dict[bias_key].to(device) """


# load my W and b
my_ckp_path = './simCLR_with_linear_layer_logs/lightning_logs/version_10/checkpoints/fixed_simCLR_linear_layer_trained-epoch=94-val_acc=0.420.ckpt'
my_ckp = torch.load(my_ckp_path)['state_dict']
my_weight = my_ckp['final_linear_layer.weight'].to(device)
my_bias = my_ckp['final_linear_layer.bias'].to(device)

""" w_diff = torch.sum((my_weight - weight)**2) / torch.numel(my_weight) 
bias_diff = torch.sum((my_bias-bias)**2) / torch.numel(my_bias)

print('sum squared diffs:')
print(w_diff)
print(bias_diff) """


#Load model
from models.SSL_linear_classifier import Encoder
#simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
simclr = Encoder(model='simCLR', path=weight_path)

""" print('OG simclr')
summary(simclr, input_size=(1, 3, 32, 32), row_settings=("depth","var_names"), depth= 10)

print('My encoder simclr')
summary(simclr, input_size=(1, 3, 32, 32), row_settings=("depth","var_names"), depth= 10) """

simclr.to(device)
simclr.freeze()

# Data
mean=[x / 255.0 for x in [125.3, 123.0, 113.9]]
std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
from lib.Get_dataset import CIFAR10_module
dataset = CIFAR10_module(mean, std, batch_size=128, augment_train=False, train_transforms=None)
dataset.prepare_data()
dataset.setup()

def calc_accuracy(my_weight, my_bias):
    # predict using the linear model in checkpoint
    correct = torch.zeros(1, device=device)
    count = torch.zeros(1, device=device)

    for batch in tqdm(dataset.val_dataloader()):
        x,y = batch[0].to(device), batch[1].to(device)

        #get features
        features = simclr(x)
        #predict 
        logits = torch.mm(features, my_weight.t().contiguous()) + torch.unsqueeze(my_bias, 0)

        _, preds  = torch.max(logits, dim=1)

        correct += torch.sum(preds==y)

        count += batch[1].size(0)
        

    return correct/count



# train a linear layer 
import torch.optim as optim

#create the linear layer
my_lin_layer = torch.nn.Linear(512, 10, bias=True).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(my_lin_layer.parameters(), lr=0.01)


#logging 
writer = SummaryWriter('linear_debug_simclr')


for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0

    pbar = tqdm(enumerate(dataset.train_dataloader(), 0))
    for i, batch in pbar:
        # get the inputs; data is a list of [inputs, labels]
        x,y = batch[0].to(device), batch[1].to(device)
        #get features
        features = simclr(x)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = my_lin_layer(features)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()


        running_loss += loss.item()

    with torch.no_grad():    
        acc = calc_accuracy(my_lin_layer.weight, my_lin_layer.bias)
    pbar.set_description(f'epoch loss: {running_loss}, epoch acc: {acc}.')

    writer.add_scalar('training loss', running_loss, epoch)
    writer.add_scalar('training acc', acc, epoch)


print('Finished Training')

print('Val acc:')
print(calc_accuracy(my_lin_layer.weight, my_lin_layer.bias) )


""" my_weight = my_lin_layer.weight
my_bias = my_lin_layer.bias """









'''for name, module in simclr.named_children():
    print()
    print(name)
    print(module)'''
 




