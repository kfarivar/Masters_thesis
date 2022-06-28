from sympy import var, Matrix
import numpy as np
import torch

from make_T_L_dataset import make_DS
from lt_experiments import Empty_model, make_deterministic


def determin_epsilon_threshold(model_path, std_images, std_labels):
    std_trained_model = Empty_model(pattern_size=3, data_mean=0, data_std=1)
    std_trained_model.load_state_dict(torch.load(model_path))
    std_trained_model.eval()
    
    preds, l_idxes, _t_idxes = std_trained_model.predict(std_images)

    for pred, l_idex, t_idx  in zip(preds, l_idxes, _t_idxes):
        print()
        print("preds (L, T): ", pred)
        print("max idx (L,T): ", (l_idex, t_idx))

    print("L filter:")
    print(std_trained_model.conv1.weight[0][0].numpy())
    print("T filter: ")
    print(std_trained_model.conv1.weight[1][0].numpy())

    L_filter = Matrix(std_trained_model.conv1.weight[0][0].numpy().flatten())
    T_filter = Matrix(std_trained_model.conv1.weight[1][0].numpy().flatten())

    # the images are in a specific order 
    # L 0th position, T 0th position, ..., T 3rd position
    # and I have checked that the correct position is maximum for conv model
    #[0, 1]
    #[2, 3]
    # create potential advresarial images using the assumptions in thesis.
    adv_images = []
    pattern_locations = [0,0, 1,1, 2,2, 3,3]
    for idx, (image, label, pat_loc) in enumerate(zip(std_images.numpy(), std_labels.numpy(), pattern_locations)):
        
        if label == 0:
            # L pattern
            # so make all 4 targeted T perturbations
            # since we have the label L we only move the T filter
            # and compare overlapping pixels ()
            for i in range(2):
                for j in range(2):





        print()
        print(label)
        print(image)
        for l_val, t_val in zip(L_filter, T_filter):
            pass


    """ 
    if original_class == 'L':
        
        return image.dot(L_filter) - image.dot(T_filter)
    else:
        return image.dot(T_filter) - image.dot(L_filter) """



if __name__ == "__main__":

    make_deterministic()
    
    with torch.no_grad():
        var('pert')

        std_images, std_labels, _, _ = make_DS(include_epsilon_bounadry=False)

        determin_epsilon_threshold('./LT_checkpoints/model.pt', std_images.unsqueeze(dim=1), std_labels)