import numpy as np
import torch
import random
from pytorch_lightning import seed_everything
from torch import cartesian_prod
from pathlib import Path
from tqdm import tqdm
import shutil
import os

def make_deterministic():
    # to make everything deterministic. 
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(0, workers=True)

def make_dataset_directory(root_path):
    # create directories 
    if os.path.isdir(root_path):
        choice = input("folders already exist do you want to overwrite [y/n]?")
        if choice == 'y':
            # remove old folder
            shutil.rmtree(root_path)
        elif choice == 'n':
            print("nothing changed.")
            raise KeyboardInterrupt()
        else:
            raise ValueError('Invalid choice !') 
    Path(f'{root_path}/L').mkdir(parents=True, exist_ok=False)
    Path(f'{root_path}/T').mkdir(parents=True, exist_ok=False)

def save_dataset_to_file(root_path, images, labels, group_size, centering, scaling):
    '''
    This is a save function that is designed to work with DatasetFolder class from pytorch.

    root_path: main path to save the dataset in subfolders named after labels 
    images: should have dimesnions : batch, hight, width
    labels: a 1D list

    These parameetrs are for reducing disk size of the dataset:
        group_size: number of images to put in a single file. Note the number of images should be divisible by group_size.
        centering, scaling: fit pixels in 8 signed bits, depends on the value of the epsilon.
    '''
    # saving multiple images as a group in a single file can reduce disk size significantly
    # first filter them according to the lable so all images in the same file have the same label
    L_images = images[labels == 0]
    T_images = images[labels == 1]
    # group images
    L_images = L_images.reshape(-1, group_size, L_images.shape[1], L_images.shape[2])
    T_images = T_images.reshape(-1, group_size, T_images.shape[1], T_images.shape[2])

    image_classes = [L_images, T_images]
    class_labels = [0, 1]

    for img_cls, label in zip(image_classes, class_labels):
        print("saving label: ", label)
        if label ==0:
            folder = f'{root_path}/L'
        elif label == 1:
            folder = f'{root_path}/T'

        for idx, img_group in tqdm(enumerate(img_cls)):
            # mult each image by scaling factor to make it fit in 8 bit signed int.
            # cloning is crucial to reduce memory usage !
            img_group = ((img_group-centering)*scaling).type(torch.int8).clone() 
            torch.save(img_group, f'{folder}/{idx}.pt')



def make_DS(include_epsilon_bounadry=False, epsilon=0.4, random_sample_boundary=False, 
            calculate_normalization=0, symmetric_options=True, perturb_pattern_pixels_only=False, exclude_std_images=False):  
    '''make the boundary L vs T pattern toy DS.
    The canvas is 4x4 and the patterns 3x3.

    For each standard image (if include_epsilon_bounadry==True) I include all the images that can be created using the advresarial perturbation,
    I will wither apply no change to a pixel xor +epsilon if it is 0 xor -epsilon if it is 1. 
    (keep the potential davresarial examples in the [0,1] bound due to my intuition.)


    include_epsilon_bounadry: whether to include extra epsilon ball boundary samples in the dataset according to l-infinity norm. 
    epsilon: the advresarial perturbation size for l-infinity.
    random_sample_boundary: (include_boundary must be True to be used) 
        False: use either 0 or epsilon when choosing pixel options
        True: use either 0 or a random perturbation samples uniformly from (0,epsilon]

    calculate_normalization: whether to calcaulate normalization factors.
        ==1: normalize pixels using all pixels mean and std
        ==2: normalize each pixel using the corresponding pixels in other images max and min
    
    symmetric_options: 
        True: include both -epsilon and +epsilon for both 0 and 1 pixels (in addition to 0)
        False: include only +epsilon for 0 pixels and only -epsilon for 1 pixels. (in addition to 0)

    perturb_pattern_pixels_only:
        True: only perturb the pixels in the 3x3 pattern and not the whole 4x4 canvas
        False: perturb all pixels on the canvas

    exclude_std_images:
        True: exclude the 0 option for pixels (i.e. the standard images will not be in the dataset.)

    Each image only contains one of the patterns which can be shifted inside the canvas !

    '''

    # the size of one side of the square canvas.
    canvas_size = 4
    # the size of one side of the square pattern.
    pattern_size = 3

    

    L_pattern = torch.tensor([ 
                [1, 0, 0],
                [1, 0, 0],
                [1, 1, 1]
            ],
            dtype=torch.float32 )

    T_pattern = torch.tensor([
                [1, 1, 1],
                [0, 1, 0],
                [0, 1, 0]
            ],
            dtype=torch.float32 )

    # The patterna and labels should match one to one
    # label 0 is L_pattern, 1 T_pattern
    std_patterns = [L_pattern, T_pattern]
    std_pattern_labels = [0,1]

    if perturb_pattern_pixels_only and include_epsilon_bounadry:
        print("perturbing only pattern pixels: ")    
        # add the potential adv patterns to standard patterns
        # Assume the image is flattened, 
        # I first create a list of set of options for each pixel: if pixel==0: {0,+epsilon} elif pixel==1: {0,-epsilon} else: error
        # then I take the cartisian product of all the sets and reshape it back to the shape of original image
        # then I add the original image to the created pattern. 
        # In this process the standard patterns are also created so no need to add them.
        patterns= []
        pattern_labels = []
        for std_pattern, std_label in zip(std_patterns, std_pattern_labels):

            if random_sample_boundary:
                # randomly choose a value in [0, epsilon) and use it instead of epsilon in the options
                # To make the comparison more fair I sample from [0.9*epsilon, epsilon)
                random_boundary = np.random.default_rng().uniform(low=0.8*epsilon, high=epsilon)
                print("random boundary is: ", random_boundary)
                list_of_options = [torch.tensor([0, random_boundary]) if pixel==0.0 else torch.tensor([0, -random_boundary]) for pixel in std_pattern.flatten()]

            else:
                print("deterministic epsilon = : ", epsilon)
                if symmetric_options: 
                    print("symmetric neighbours:")
                    list_of_options = [torch.tensor([0.0, epsilon, -epsilon]) if pixel==0.0 else torch.tensor([0.0, -epsilon, epsilon]) for pixel in std_pattern.flatten()]
                else: 
                    list_of_options = [torch.tensor([0.0, epsilon]) if pixel==0.0 else torch.tensor([0.0, -epsilon]) for pixel in std_pattern.flatten()]

            cartesian_of_options = cartesian_prod(*list_of_options)

            for option in cartesian_of_options:
                adv_image = option.reshape(std_pattern.shape[0], std_pattern.shape[1])
                adv_pattern = std_pattern + adv_image
                patterns.append(adv_pattern)
                pattern_labels.append(std_label)


    else: 
        patterns = std_patterns
        pattern_labels = std_pattern_labels



    dataset_list = []
    final_labels = []
    # move the images to all the 4 locations
    for i in range(2):
        for j in range(2):
            for pattern, label in zip(patterns, pattern_labels):
                canvas = torch.zeros(canvas_size, canvas_size, dtype=torch.float)
                canvas[i:i+pattern_size, j:j+pattern_size] = pattern

                 
                if not perturb_pattern_pixels_only and include_epsilon_bounadry and symmetric_options:
                    # if we want to perturb all canvas pixels we do it here
                    # note the standard image is one of the options here
                    print("preturbing all canvas pixels symmetrically.")
                    if exclude_std_images:
                        list_of_options = [torch.tensor([ epsilon, -epsilon]) if pixel==0.0 else torch.tensor([-epsilon, epsilon]) for pixel in canvas.flatten()]
                    else:
                        list_of_options = [torch.tensor([0.0, epsilon, -epsilon]) if pixel==0.0 else torch.tensor([0.0, -epsilon, epsilon]) for pixel in canvas.flatten()]
                    
                    cartesian_of_options = cartesian_prod(*list_of_options)

                    # all possible potential advresarial images
                    all_adv_perturbations = cartesian_of_options.reshape(cartesian_of_options.shape[0], canvas_size, canvas_size)
                    # broadcast canvas addition
                    all_adv_images = canvas + all_adv_perturbations
                    dataset_list.append(all_adv_images)
                    final_labels.append( torch.full((all_adv_images.shape[0],), label) ) # making them tensor here makes it faster
                    
                
                elif not perturb_pattern_pixels_only and not include_epsilon_bounadry:
                    print("only adding standard images in the dataset.")
                    # add a dim for batch concat 
                    canvas = torch.unsqueeze(canvas ,0)
                    dataset_list.append(canvas)
                    final_labels.append(torch.tensor([label]))
                
                else:
                    raise NotImplementedError("your dataset creation parameters are not implemenetd.")

                

    
    #then put them in 1 batch
    final_dataset = torch.concat(dataset_list, dim=0)
    final_labels = torch.concat(final_labels)
    assert final_labels.size(0) == final_dataset.size(0)
    

    # default values no normalization
    mean = torch.tensor([0])
    std = torch.tensor([1])
    if calculate_normalization == 1:
        mean = final_dataset.mean() 
        std = final_dataset.std()
        print(f"Data normalized with all pixels mean {mean} and std {std}")

    elif calculate_normalization == 2:
        raise NotImplementedError("normalization not implemented")
        """ print("Data normalized with each pixels corresponding min and max")
        normalized_data = final_dataset.clone()
        normalized_data = normalized_data.view(final_dataset.size(0), -1)
        normalized_data -= normalized_data.min(1, keepdim=True)[0]
        normalized_data /= normalized_data.max(1, keepdim=True)[0]
        normalized_data = normalized_data.view(final_dataset.size(0), final_dataset.size(1), final_dataset.size(2))
        final_dataset = normalized_data """

    # add the channel dimension, note that mean can be min and std can be max depending on the type of normlaization
    return final_dataset, final_labels, mean, std


    


if __name__ == '__main__':
    make_deterministic()
    # dataset size is 3^16 * 8 group size should be a multiple
    group_size = 2**6
    include_epsilon_bounadry= True
    epsilon = 0.41
    scaling = 100
    centering = 0.5
    exclude_std_images = True
    dataset_root_folder = f'L_T_dataset_groupsize_{group_size}_include_boundary{include_epsilon_bounadry}_epsilon{epsilon}_center_{centering}_scale_{scaling}_exclude_std_images{exclude_std_images}'
    
    # do this first, don't pause the dataset creation !
    make_dataset_directory(dataset_root_folder)

    images, labels, _, _ = make_DS(include_epsilon_bounadry=include_epsilon_bounadry, epsilon=epsilon, exclude_std_images=exclude_std_images)
    
    print("total number of images: ", images.shape)
    
    save_dataset_to_file(dataset_root_folder, images, labels, group_size, centering, scaling)