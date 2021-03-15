from AdvLib import Adversarisal_bench as ab

#create a pytorch dataloader for test set
test_set = None

# initialize an attack
fgsm = FGSM(self.model, self.loss, epsilon)

# get robust accuracy
acc, samples = ab.measure_models_robust_accuracy(test_set, [fgsm])

# get the dataset concentration
all_samples = None
concent = ad.measure_dataset_concentration(all_samples)

# get an only non-robust dataset
train_set = None
non_robuts_train = ad.make_non_robust_dataset(train_set)

# make a robust dataset ?????

