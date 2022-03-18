"""The pyraug's Datasets inherit from
:class:`torch.utils.data.Dataset` and must be used to convert the data before
training. As of today, it only contains the :class:`pyraug.data.BaseDatset` useful to train a
VAE model but other Datatsets will be added as models are added.
"""
import os
import numpy  as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, utils
from collections import defaultdict

torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BarrettLabeledDataset(Dataset):
    
    def __init__(self, root_dir, transforms_=None):
        super().__init__()
        
        self.root_dir = root_dir
        self.transforms_ = transforms_
        datasets = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        for dataset in datasets:
            biopsypath = os.path.join(root_dir, dataset)
            self.biopsies = os.listdir(biopsypath)
            self.patches = defaultdict(lambda : [])

            self.patchIDs = []
            for biopsy in self.biopsies:
                subpath = os.path.join(biopsypath, biopsy)
                self.subbio = os.listdir(subpath)

                for subbiopsy in self.subbio:
                    biopsy_patches = os.path.join(subpath, subbiopsy, 'biopsy_patches')
                    
                    self.patches[subbiopsy] = os.listdir(biopsy_patches)


                    [self.patchIDs.append((dataset, subbiopsy, patch)) 
                            for patch in self.patches[subbiopsy]]



    def __getitem__(self, index):

        dataset, biopsy, patch_id = self.patchIDs[index]

        patch_path = os.path.join(*[self.root_dir, dataset, biopsy[:-2], biopsy, 'biopsy_patches', patch_id])
        mask_path = os.path.join(*[self.root_dir, dataset, biopsy[:-2], biopsy, 'mask_patches', patch_id])

        patch = self.transforms_['biopsy_patches'](Image.open(patch_path))
        mask = np.array(Image.open(mask_path))  
        
        area_in_pixels = np.bincount(mask.flatten(), minlength=6)

#         # We currently get Squamous, NDBE, LGD and HG as classes, but can be changed in the future
        self.patch_dict = {  0: area_in_pixels[2].item(),
                        1 : area_in_pixels[3].item(),
                        2 : area_in_pixels[4].item(),
                        3 : area_in_pixels[5].item(),
                    }
        label = max(self.patch_dict, key=self.patch_dict.get)       

        return patch, label

    def __len__(self):
        return len(self.patchIDs)

    def N_patches(self):
        return len(self)

    def N_patches_biopsy(self, biopsy):
        return len(self.patches[biopsy])


# MAIN CODE
transforms_ = {'biopsy_patches' : transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])}

training_data = BarrettLabeledDataset(root_dir="datasplit/train/", transforms_=transforms_)
test_data = BarrettLabeledDataset(root_dir='datasplit/test/', transforms_=transforms_)


def get_class_distribution(dataset_obj):
    count_dict = {k:0 for k in range(4)}
    for element in dataset_obj:
        y_lbl = element[1]
        count_dict[y_lbl] += 1  
    return count_dict

# TODO: store results of get_class_distribution in .txt file, 
# load from text file instead of recomputing if file exists. 
counts = get_class_distribution(training_data)
class_count = [i for i in counts.values()]

class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
y_train = np.array([sample[1] for sample in training_data])
samples_weight = np.array([class_weights[t] for t in y_train])
final_weights = torch.from_numpy(samples_weight)

# WEIGHTED DATALOADER OBJECT
sampler = WeightedRandomSampler(final_weights.type('torch.DoubleTensor'), len(final_weights))
training_loader = DataLoader(training_data, batch_size=12, sampler=sampler)
