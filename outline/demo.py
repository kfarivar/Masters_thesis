import torch

#from AdvLib import Adversarisal_bench as ab

from Get_dataset import get_cifar

from simple_model import Net

from Normal_training import non_robust_training



#create a pytorch dataloader for test set
trainloader,testloader, classes = get_cifar()
#use GPU
device = 'cpu' # torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print('Device is:' , device)
# get model
model = Net().to(device)
# train the model normally (non_robust)
model = non_robust_training (model, trainloader, epochs=10, device=device)




# #initialize and send the model to AdvLib
# simple_model_bench = ab(model, use_cuda=True)

# # initialize an attack
# fgsm = FGSM(self.model, self.loss, epsilon)

# # get robust accuracy
# acc, samples = ab.measure_models_robust_accuracy(test_set, [fgsm])

# # get the dataset concentration
# all_samples = None
# concent = ad.measure_dataset_concentration(all_samples)

# # get an only non-robust dataset
# train_set = None
# non_robuts_train = ad.make_non_robust_dataset(train_set)

# # make a robust dataset ?????

