#import faiss
import torchvision
import torch.utils.data
import numpy as np
import os
from typing import Callable, Optional
from spaces import NBoxSpace
import torchvision.transforms.functional as TF
    
class CausalDataset(torch.utils.data.Dataset):
    """Load Causal3DIdent dataset"""
    def __init__(self, classes, root: str, 
                    transform: Optional[Callable] = None, 
                    tuning_transforms=None,
                    loader: Optional[Callable] = torchvision.datasets.folder.pil_loader,
                    latent_dimensions_to_use=None, 
                    biaugment=False, 
                    use_augmentations=False, 
                    approximate_mode: Optional[bool] = True, 
                    change_all_positions=False, 
                    change_all_hues=False, 
                    change_all_rotations=False, 
                    apply_rotation=False,
                    include_index=False
    ):
        '''
        biaugment: if True return 2 augmented versions of the same image (like simCLR), otherwise return a single image.
        use_augmentations: if True applies random rotation (0,90, 180, 270) then the trasform sent.

        tuning_transforms: get the sample needed for online fine tuning linear layer in my SSL code

        include_index: whether to include image index for each batch.

        The images are rendered based on a 10-dimensional latent, where: (i) three dimensions describe the XYZ position of the object, (ii)
        three dimensions describe the rotation of the object in Euler angles, (iii) two dimensions describe the 
        colour (hue) of the object and the ground of the scene, respectively, and (iv) two dimensions describe the position and colour (hue) of the spotlight.

        
        render internally assumes the variables form these value ranges:
        
        per object:
            0. x position in [-3, -3]
            1. y position in [-3, -3]
            2. z position in [-3, -3]
            3. alpha rotation in [0, 2pi]
            4. beta rotation in [0, 2pi]
            5. gamma rotation in [0, 2pi]
            6. theta spot light in [0, 2pi]
            7. hue object in [0, 2pi]
            8. hue spot light in [0, 2pi]
        
        per scene:
            9. hue background in [0, 2pi]
    

        '''
        super(CausalDataset, self).__init__()
        self.apply_rotation = apply_rotation
        self.change_list = []
        self.change_all_positions = change_all_positions
        if self.change_all_positions:
            self.change_list += [0,1,2,-4]
        self.change_all_hues = change_all_hues
        if self.change_all_hues:
            self.change_list += [-1,-2,-3]
        self.change_all_rotations = change_all_rotations
        if self.change_all_rotations:
            self.change_list += [3,4,5]
        self.use_augmentations = use_augmentations
        self.space = NBoxSpace(n=1, min_=-1., max_=1.)
        self.sigma = 1.0
        self.root = root
        self.biaugment = biaugment
        self.classes = classes
        self.latent_classes = []
        self.tuning_transforms = tuning_transforms
        self.include_index = include_index

        # read the latent values for each image in each class
        for i in classes:
            self.latent_classes.append(np.load(os.path.join(root, 
                                                        "raw_latents_{}.npy".format(i))))
        self.unfiltered_latent_classes = self.latent_classes

        if latent_dimensions_to_use is not None:
            for i in classes:
                self.latent_classes[i] = np.ascontiguousarray(self.latent_classes[i][:,
                                                                latent_dimensions_to_use])

        if transform is None: transform = lambda x: x
        self.transform = transform
        self.image_paths_classes = []

        # paths of images to be read
        for i in classes:
            max_length = int(np.ceil(np.log10(len(self.latent_classes[i]))))
            self.image_paths_classes.append([os.path.join(root, "images_{}".format(i), 
                                            f"{str(j).zfill(max_length)}.png") for j in range(self.latent_classes[i].shape[0])])
        self.loader = loader
        """ if not self.use_augmentations:
            self._index_classes = []
            for i in classes:
                if approximate_mode:
                    # faiss is a library for handeling big vectors in gpu, here the latents vector. (not sure how it works)
                    # If they want to do augmentation using the latents they 1.add random variation to latents 
                    # 2.use this library to serach for the closest sample according to euclidean norm.  
                    _index = faiss.index_factory(self.latent_classes[i].shape[1], 
                                                 "IVF1024_HNSW32,Flat")
                    _index.efSearch = 8
                    _index.nprobe = 10
                else:
                    _index = faiss.IndexFlatL2(self.latent_classes[i].shape[1])

                if approximate_mode:
                    _index.train(self.latent_classes[i])
                _index.add(self.latent_classes[i])
                self._index_classes.append(_index) """

    def __len__(self) -> int:
        return len(self.latent_classes[0]) * len(self.classes)

    def __getitem__(self, item):
        z = self.latent_classes[item // len(self.latent_classes[0])][item % len(self.latent_classes[0])]
        path_z = self.image_paths_classes[item // len(self.latent_classes[0])][item % len(self.latent_classes[0])]

        # read the image
        sample = self.loader(path_z)
        if self.apply_rotation:
            angles = np.random.choice([0, 90, 180, 270], size=2, replace=False)
            x1 = self.transform(TF.rotate(sample, int(angles[0])))
        else:
            x1 = self.transform(sample)


        if self.biaugment:
            if self.use_augmentations:
                z_tilde = z
                if self.apply_rotation:
                    x2 = self.transform(TF.rotate(sample, int(angles[1])))
                else:
                    x2 = self.transform(sample)
            """ 
            else:
                z_tilde = np.copy(z)
                for j in self.change_list:
                    z_tilde[j] = self.space.normal( torch.reshape(torch.from_numpy(np.array([z[j]])) , (1,1)), 
                                                   self.sigma, size=1, device="cpu").numpy().flatten()
                # I think this returns the 2 closests vectors (Euclidean), 
                # since the augmentation is random z might still be closest to z_tilda so we get the 2 closest and choose the different one. 
                _, index_z_tilde = self._index_classes[item // len(self.latent_classes[0])].search(z_tilde[np.newaxis], 2)
                if index_z_tilde[0, 0] != (item % len(self.latent_classes[0])):
                    index_z_tilde = index_z_tilde[0, 0]
                else:
                    index_z_tilde = index_z_tilde[0, 1]
                z_tilde = self.latent_classes[item // len(self.latent_classes[0])][index_z_tilde]
                path_z_tilde = self.image_paths_classes[item // len(self.latent_classes[0])][index_z_tilde]
                x2 = self.transform(self.loader(path_z_tilde)) """
            # get the sample needed for online fine tuning linear layer
            tune_sample = self.tuning_transforms(sample)
            # simclr uses the first two images for training or validation of unsupervised loss and the third image for training the online readout.
            #  item // len(self.latent_classes[0]) is object class.
            return (x1, x2, tune_sample), item // len(self.latent_classes[0]) #, (z.flatten(), z_tilde.flatten())
        else:
            # This is used during finetuning or supervised learning.
            if self.include_index:
                return  x1, item // len(self.latent_classes[0]) , z.flatten(), item
            else:
                return  x1, item // len(self.latent_classes[0]) , z.flatten()