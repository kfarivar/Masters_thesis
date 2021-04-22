import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional

class CIFAR10_module(pl.LightningDataModule):

    def __init__(self, mean, std, data_dir: str = "../data" , batch_size: int = 32, num_workers=12):
        '''The mean and std used for normalization during training should be sent.
        '''
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean, std)])


    def prepare_data(self):
        # download
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        # Assign train dataset for use in dataloaders
        if stage == 'fit' or stage is None:
            self.cifar10_train = CIFAR10(root=self.data_dir, train=True, download=True, transform=self.transforms)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.cifar10_test = CIFAR10(root=self.data_dir, train=False, download=True, transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

