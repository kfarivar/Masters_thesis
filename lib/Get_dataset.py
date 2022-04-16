from select import select
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional

def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.

    e.g use:
    
    MNISTWithIndices = dataset_with_indices(MNIST)
    dataset = MNISTWithIndices('~/datasets/mnist')

    for batch_idx, (data, target, idx) in enumerate(loader):
        print('Batch idx {}, dataset index {}'.format(
            batch_idx, idx))

    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


class CIFAR10_module(pl.LightningDataModule):
    ''' Note the Train is shuffeled but not the test
    '''

    def __init__(self, mean, std, data_dir: str = "./data" , batch_size: int = 32, num_workers=12, augment_train=True, 
                train_transforms=None):
        '''The mean and std used for normalization during training should be sent.
           augment_train: if True applies default data augmentation (randomcrop, horizontal flip) for train set.
           train_transforms: apply custom transform to train set. (ignores default augmentation and normalization !)

        '''
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = mean
        self.std = std
        # we might need individual indices
        self.cifar_with_idx = dataset_with_indices(CIFAR10)
        self.augment_train = augment_train
        self.train_trans = train_transforms


    def prepare_data(self):
        # download
        self.cifar_with_idx(root=self.data_dir, train=True, download=True)
        self.cifar_with_idx(root=self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        # Assign train dataset for use in dataloaders
        if stage == 'fit' or stage is None:

            if self.train_trans is not None:
                train_transforms = self.train_trans
                                                 

            elif self.augment_train:
                train_transforms = transforms.Compose(
                                            [transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std) ])
            else:
                train_transforms = transforms.Compose(
                                            [transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std) ])

            self.cifar10_train = self.cifar_with_idx(root=self.data_dir, train=True, download=True, transform=train_transforms)

        
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            test_transforms = transforms.Compose(
                                        [transforms.ToTensor(),
                                        transforms.Normalize(self.mean, self.std)])
            self.cifar10_test = self.cifar_with_idx(root=self.data_dir, train=False, download=True, transform=test_transforms)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, shuffle=True, 
                         num_workers=self.num_workers, pin_memory=True)
                         

    def val_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, shuffle=False, 
                        num_workers=self.num_workers, pin_memory=True)

    @property
    def num_samples(self) -> int:
        
        return 50_000 #train_len

    @property
    def num_classes(self) -> int:
        """
        Return:
            10
        """
        return 10


from PIL import Image 
import numpy as np                        
import sys
sys.path.append('/home/kiarash_temp/adversarial-components/3dident_causal/kiya_3dident')
from clevr_dataset import CausalDataset



class Causal_3Dident(pl.LightningDataModule):
    '''
    used for finetuning SSL or supervised training.
    There is no color jitter. random crops only applied for supervised.
    '''

    def __init__(self, data_dir, augment_train=False, no_normalization=False, batch_size: int = 32, 
                    num_workers=32, train_subset=1, val_subset=1, train_include_index=False, val_include_index=False):
        '''
        [train/val]_subset: if ratio between 0 and 1 chooses a random subset of data for train/eval of size ratio*len(dataset).

        no_normalization: if true the images are normalized between [0,1]. if False, the original data normalization is applied.

        include_index: whether to include index of samples in each batch.
        '''
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = 7
        self.augment_train = augment_train
        self.train_subset = train_subset
        self.val_subset = val_subset  
        self.no_normalization = no_normalization
        self.train_include_index = train_include_index
        self.val_include_index = val_include_index

    def setup(self, stage: Optional[str] = None):
        mean_per_channel = [0.4327, 0.2689, 0.2839]
        std_per_channel = [0.1201, 0.1457, 0.1082]
        transform_list = []

        # random crop (0.08 is considered small crop in paper.)
        supervised_transforms = [transforms.RandomResizedCrop(224, 
                                scale=(0.08, 1.0), 
                                interpolation=Image.BICUBIC), 
                            transforms.RandomHorizontalFlip()]

        if self.no_normalization:
            basic_transforms = [transforms.ToTensor()]
        else:
            basic_transforms = [transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=mean_per_channel,
                                        std=std_per_channel
                                    )]


        if self.augment_train:                     
            transform_list = supervised_transforms + basic_transforms

        else:
            transform_list = basic_transforms

        dataset_kwargs = {'transform':transforms.Compose(transform_list)}

        latent_dimensions_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        dataset_kwargs["latent_dimensions_to_use"] = latent_dimensions_to_use

        # the dataset
        self.train_dataset = CausalDataset(
                                        classes=np.arange(7), 
                                        root='{}/trainset'.format(self.data_dir), 
                                        biaugment=False,
                                        use_augmentations=False, # doesn't matter since we dont use biaugment. (trasforms are still applied)
                                        change_all_positions=False,
                                        change_all_hues=False,
                                        change_all_rotations=False,
                                        apply_rotation=False,
                                        include_index = self.train_include_index,
                                        **dataset_kwargs # this is where we send the transformations.
                                        ) 

        # val data (same as test)
        dataset_kwargs['transform'] = transforms.Compose(basic_transforms)
        self.val_dataset = CausalDataset(
                                        classes=np.arange(7), 
                                        root='{}/testset'.format(self.data_dir), 
                                        biaugment=False,
                                        use_augmentations=False, # Should be true if we apply either of crop/color distort/ rotation
                                        change_all_positions=False,
                                        change_all_hues=False,
                                        change_all_rotations=False,
                                        apply_rotation=False,
                                        include_index= self.val_include_index,
                                        **dataset_kwargs # this is where we send the transformations.
                                        ) 
        


        # select random subset of data
        def select_subset(dataset, ratio):
            if (ratio > 0) and (ratio <1):
                print('using subset of whole dataset.')
                indexes = np.random.choice(len(dataset), size=int(ratio*len(dataset)), replace=False)
                sub_dataset = torch.utils.data.Subset(dataset, indexes)
            else:
                print('using whole dataset.')
                sub_dataset = dataset
            
            return sub_dataset

        self.train_dataset = select_subset(self.train_dataset, self.train_subset)
        self.val_dataset = select_subset(self.val_dataset, self.val_subset)

        self.num_samples = len(self.train_dataset)


        
    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, 
                                                num_workers=self.num_workers,
                                                pin_memory=True, shuffle=True)
        return train_loader

    def val_dataloader(self, shuffle=False):
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, 
                                                num_workers=self.num_workers,
                                                pin_memory=True, shuffle=shuffle)
        return val_loader
        

    def test_dataloader(self, shuffle=False):
        
        return self.val_dataloader(shuffle=shuffle)