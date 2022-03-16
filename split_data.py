# File that creates two new directories and places the given datasets in train and test dirs. 
# Includes option to balance the train set by removing some of the dominant class samples from the data.

import os
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from distutils.dir_util import copy_tree


def get_classes(patch_dir, datasets):
    label_dict = {}
    for dataset in datasets:
        for WSI_name in os.listdir(os.path.join(patch_dir, dataset)):
            for biopsy in os.listdir(os.path.join(*[patch_dir, dataset, WSI_name])):
                for number in os.listdir(os.path.join(*[patch_dir, dataset, WSI_name, biopsy])):
                    for b, m in zip(os.listdir(os.path.join(*[patch_dir, dataset, WSI_name, biopsy, 'biopsy_patches'])), 
                                    os.listdir(os.path.join(*[patch_dir, dataset, WSI_name, biopsy, 'mask_patches']))):
                        bpath = os.path.join(*[patch_dir, dataset, WSI_name, biopsy, 'biopsy_patches', b])
                        mpath = os.path.join(*[patch_dir, dataset, WSI_name, biopsy, 'mask_patches', m])

                        mask = np.array(Image.open(mpath))  
                        area_in_pixels = np.bincount(mask.flatten(), minlength=6)
                        patch_dict = {  'Background': area_in_pixels[0].item(),
                                        'Stroma': area_in_pixels[1].item(),
                                        'Squamous': area_in_pixels[2].item(),
                                        'NDBE' : area_in_pixels[3].item(),
                                        'LGD' : area_in_pixels[4].item(),
                                        'HGD' : area_in_pixels[5].item(),
                                    }

                        label_dict[bpath] = patch_dict
    return label_dict

def remove_dominant_class(class_dict, threshold):
    df = pd.DataFrame.from_dict(class_dict, orient='index')
    df.index.name = 'Patch Path'
    fig, ax = plt.subplots()
    ax.pie(list(df.sum()), labels=list(df), autopct='%1.1f%%')
    fig.savefig('pie_before.png')
    dominant_class = max([(c,l) for c,l in zip(list(df.sum()), list(df))])[1]
    print('Dominant class in dataset is:', dominant_class)

    percentages = df.div(df.sum(axis=1), axis=0) 
    newdf = percentages.drop(percentages[(percentages[dominant_class] > threshold)].index)
    fig2, ax2 = plt.subplots()
    ax2.pie(list(newdf.sum()), labels=list(newdf), autopct='%1.1f%%')
    fig2.savefig('pie_after.png')
    return list(newdf.index.values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration of parameters for class_dist.")

    parser.add_argument('--patch_dir', type=str, default="barrett20x64",
                        help='Path to directory  containinng the patches')
    parser.add_argument('--out_dir', type=str, default='datasplit',
                        help='Path to store patches of dataset')
    parser.add_argument('--train_datasets', default=["ASL", "LANS", "RBE", "RBE_Nieuw"],
                        help='Datasets to train on')
    parser.add_argument('--test_datasets', default=["Bolero"],
                        help='Datasets to test on')    
    parser.add_argument('--threshold', type=float, default=0.9,
                        help='Threshold for determining what percentage of the dominant class in one image is acceptable')
    parser.add_argument('--balance', type=bool, default=True,
                        help='Whether to balance the data or just split')
           

    config = parser.parse_args()

    class_dict = get_classes(config.patch_dir, config.train_datasets)
    if config.balance is True:
        training_paths = remove_dominant_class(class_dict, config.threshold)
    else:
        training_paths = list(class_dict.keys())
    mask_paths = [p.replace('biopsy_patches', 'mask_patches') for p in training_paths]

    for patch, mask in zip(training_paths, mask_paths):
        os.makedirs(os.path.dirname(os.path.join(config.out_dir, "train", "/".join(patch.strip("/").split('/')[1:]))), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.join(config.out_dir, "train", "/".join(mask.strip("/").split('/')[1:]))), exist_ok=True)
        shutil.copy(patch, os.path.join(config.out_dir, "train", "/".join(patch.strip("/").split('/')[1:])))
        shutil.copy(mask, os.path.join(config.out_dir, "train", "/".join(mask.strip("/").split('/')[1:])))

    for dataset in config.test_datasets:
        os.makedirs(os.path.join(config.out_dir, 'test', dataset), exist_ok=True)
        copy_tree(os.path.join(config.patch_dir, dataset), os.path.join(config.out_dir, 'test', dataset)) 