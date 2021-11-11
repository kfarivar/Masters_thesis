''' Entities for My modifications to simCLR'''
from typing import Any, Optional, Union
from torchvision import transforms
import numpy as np

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