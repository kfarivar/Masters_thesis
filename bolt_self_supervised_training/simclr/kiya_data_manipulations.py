''' Entities for My modifications to simCLR'''
import torch
import pytorch_lightning as pl
from typing import Any, Optional, Union
from torchvision import transforms
import numpy as np
from PIL import Image 
        

from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform


class CIFAR10DataModule_class_pairs(CIFAR10DataModule):
    '''
    Warning: Don't use this with small batch size, the samples without a match from the same class (other than themselves) will be droped from batch.
    This cifar data module implements the on_after_batch_transfer hook 
    which pairs images with a random image from the same class;
    Instead of two augmented versions of the same image.
    Intuitively it might make sense to use double the batch size than original simCLR since there we have more image variation due to augmentation !
    '''
    def __init__(self, data_dir: Optional[str] = None, val_split: Union[int, float] = 0.2, num_workers: int = 0, normalize: bool = False, 
                batch_size: int = 32, seed: int = 42, shuffle: bool = True, pin_memory: bool = True, drop_last: bool = False, *args: Any, **kwargs: Any):
        super().__init__(data_dir=data_dir, val_split=val_split, num_workers=num_workers, normalize=normalize, batch_size=batch_size, seed=seed, 
                        shuffle=shuffle, pin_memory=pin_memory, drop_last=drop_last, *args, **kwargs)

    
    def on_before_batch_transfer(self, batch, dataloader_idx):
        '''
        If an image in batch is single in its class it will be dropped !
        Returns:
        1. the batch of original images (augmented) 
        2. the indexes for matches randomly selected based on class
        3. val images
        4. y labels
        '''
        #if self.trainer.training or self.trainer.validating:
        # do this for all cases

        # final image in tuple is for online eval don't modify
        # I modified the transofmer so we only get a single image (other than val image)
        (img1, val_image), y = batch

        # use label to randomly find another image from the same class (exclude the image itself!)
        # and create pairs and finallly return the pairs  

        def select_image_indexes(labels):
            ''' for each image select another random image from the same class.
            returns the indexes of matching pairs. 
            '''
            labels = labels.numpy()
            indexes = np.arange(labels.shape[0])
            # we have 10 classes  
            classes = np.arange(10)
            # make a dict of class_number:array_of_indexes
            class_index = {}
            for c in classes:
                class_index[c] = indexes[labels==c]
            
            indexes1 = []
            indexes2 = []
            for idx in indexes:
                class_ = labels[idx]
                options = class_index[class_] 
                # don't pair it with itself
                options = options[options!= idx]
                
                # if this is the only sample from this class drop it from batch !
                if options.size != 0:
                    indexes1.append(idx)
                    # randomly select one of options
                    idx2 = np.random.choice(options, 1)[0]
                    indexes2.append(idx2)
        
            return indexes1, indexes2

        idx1, idx2 = select_image_indexes(y)

        img1_selected = img1[idx1]
        img2_seleted = img1[idx2]
        # I also exclude the same images in val (don't think this really matters)
        val_image_selected = val_image[idx1]
        y_selected = y[idx1]

        return (img1_selected, img2_seleted, val_image_selected) , y_selected


class single_images_train_transform(SimCLRTrainDataTransform):
    '''
    Tranform used for my case of pairing images from the same class
    we save time by not doing the extra transform on the second.
    I overwrite the __call__ function.
    '''

    def __init__(self, input_height: int = 224, gaussian_blur: bool = True, jitter_strength: float = 1, normalize=None):
        super().__init__(input_height=input_height, gaussian_blur=gaussian_blur, jitter_strength=jitter_strength, normalize=normalize)

    def __call__(self, sample):
        xi = self.train_transform(sample)
        return xi, self.online_transform(sample)

    
class single_images_val_transform(single_images_train_transform):
    '''similar to SimCLREvalDataTransform'''

    def __init__(
        self, input_height: int = 224, gaussian_blur: bool = True, jitter_strength: float = 1.0, normalize=None
    ):
        super().__init__(
            normalize=normalize, input_height=input_height, gaussian_blur=gaussian_blur, jitter_strength=jitter_strength
        )

        # replace online transform with eval time transform
        self.online_transform = transforms.Compose(
            [
                transforms.Resize(int(self.input_height + 0.1 * self.input_height)),
                transforms.CenterCrop(self.input_height),
                self.final_transform,
            ]
        )


class CIFAR10_use_all_train(CIFAR10DataModule):
    '''The original DS doesn't use all the train this one uses all 50_000 train images.'''

    def __init__(
        self,
        data_dir: Optional[str] = None,
        num_workers: int = 0,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: Where to save/load the data
            val_split: Percent (float) or number (int) of samples to use for the validation split
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(  # type: ignore[misc]
            data_dir=data_dir,
            val_split= 50_000,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            *args,
            **kwargs,
        )


    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""
        if stage == "fit" or stage is None:
            train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
            val_transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms

            dataset_train = self.dataset_cls(self.data_dir, train=True, transform=train_transforms, **self.EXTRA_ARGS)
            
            dataset_val = self.dataset_cls(self.data_dir, train=False, transform=val_transforms, **self.EXTRA_ARGS)

            # Split (changed !)
            self.dataset_train = dataset_train
            self.dataset_val = dataset_val

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
            self.dataset_test = self.dataset_cls(
                self.data_dir, train=False, transform=test_transforms, **self.EXTRA_ARGS
            )
    
    def _get_splits(self, len_dataset: int):
        return None

    def _split_dataset(self, dataset, train: bool = True) :
        return None
    
    @property
    def num_samples(self) -> int:
        return 50_000

    @property
    def num_classes(self) -> int:
        """
        Return:
            10
        """
        return 10

import sys
sys.path.append('/home/kiarash_temp/adversarial-components/3dident_causal/kiya_3dident')
from clevr_dataset import CausalDataset

class Causal_3Dident (pl.LightningDataModule):
    def __init__(self, data_dir, jitter_strength = 1, batch_size: int = 32, num_workers=16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        # strength of color distortion.
        self.jitter_strength = jitter_strength
        self.num_workers = num_workers
    
        self.num_classes = 7

    def setup(self, stage: Optional[str] = None):
        mean_per_channel = [0.4327, 0.2689, 0.2839]
        std_per_channel = [0.1201, 0.1457, 0.1082]
        transform_list = []

        # random crop (0.08 is considered small crop in paper.)
        transform_list += [transforms.RandomResizedCrop(224, 
                                scale=(0.08, 1.0), 
                                interpolation=Image.BICUBIC), 
                            transforms.RandomHorizontalFlip()]
        # color distortion
        color_jitter = transforms.ColorJitter(0.8*self.jitter_strength, 0.8*self.jitter_strength, 0.8*self.jitter_strength, 0.2*self.jitter_strength)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
        transform_list += [color_distort]

        transform_test_list = [transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=mean_per_channel,
                                    std=std_per_channel
                                )]
        transform_list += transform_test_list
        dataset_kwargs = dict(transform=transforms.Compose(transform_list))
        latent_dimensions_to_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        dataset_kwargs["latent_dimensions_to_use"] = latent_dimensions_to_use

        #online transforms 
        online_trans = transforms.Compose([
                                transforms.RandomResizedCrop(224, 
                                    scale=(0.08, 1.0), 
                                    interpolation=Image.BICUBIC), 
                                transforms.RandomHorizontalFlip(), 
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=mean_per_channel,
                                    std=std_per_channel
                                )])
        dataset_kwargs['tuning_transforms'] = online_trans

        # the dataset
        self.train_dataset = CausalDataset(
                                        classes=np.arange(7), 
                                        root='{}/trainset'.format(self.data_dir), 
                                        biaugment=True,
                                        use_augmentations=True, # Should be true if we apply either of crop/color distort/ rotation
                                        change_all_positions=False,
                                        change_all_hues=False,
                                        change_all_rotations=False,
                                        apply_rotation=False,
                                        **dataset_kwargs # this is where we send the transformations.
                                        ) 
        # for my simclr validation first 2 images should have the same augmentation status as train. 
        # But the third should have no augmentation (not even flip)!
        dataset_kwargs['tuning_transforms'] = transforms.Compose(transform_test_list)
        self.val_dataset = CausalDataset(
                                        classes=np.arange(7), 
                                        root='{}/testset'.format(self.data_dir), 
                                        biaugment=True,
                                        use_augmentations=True, # Should be true if we apply either of crop/color distort/ rotation
                                        change_all_positions=False,
                                        change_all_hues=False,
                                        change_all_rotations=False,
                                        apply_rotation=False,
                                        **dataset_kwargs # this is where we send the transformations.
                                        ) 

        # unused.
        dataset_kwargs['transform'] = transforms.Compose(transform_test_list)
        self.test_dataset = CausalDataset(classes=np.arange(7), 
                                    root='{}/testset'.format(self.data_dir), 
                                    biaugment=False, 
                                    **dataset_kwargs)
        self.num_samples = len(self.train_dataset)
        
        
    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, 
                                                num_workers=self.num_workers,
                                                pin_memory=True, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, 
                                                num_workers=self.num_workers,
                                                pin_memory=True, shuffle=True)
        return val_loader
        

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, 
                                                num_workers=self.num_workers,
                                                pin_memory=True, shuffle=False)
        return test_loader
