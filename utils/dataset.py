# %%
import os
import json
from random import sample, seed

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
# import cv2

class BarrettDataset(Dataset):
    def __init__(self, root_dir, labels_list=[], N_biopsies=None, transforms_=None, biopsy_patches_only=True):
        super().__init__()
        
        self.root_dir = root_dir
        self.biopsy_patches_only = biopsy_patches_only

        self.transforms_ = transforms_
        self.labels_list = labels_list
        with open(os.path.join(*[root_dir, 'meta_data.json']), 'r') as fp:
            meta_data = json.load(fp)
        self._patch_size = meta_data['patch_size']

        self.patches_frame = pd.read_csv(os.path.join(*[root_dir, 'labels_patches_level.csv']), index_col=['Dataset', 'WSI_name', 'Biopsy', 'Patch_idx'])
        if N_biopsies is not None:
            seed(1234)
            self.patches_frame = self.patches_frame.loc[slice(None), slice(None), sample(list(self.patches_frame.index.unique('Biopsy')), k=N_biopsies), slice(None)].sort_index(axis=0, level=['Dataset', 'WSI_name', 'Biopsy'], ascending=True)

    def __getitem__(self, index):
        labels = self.patches_frame.iloc[index][self.labels_list]
        patch_path = list(labels.name)

        biopsy_path = patch_path[:]
        biopsy_path.insert(-1, 'biopsy_patches')
        biopsy = Image.open(os.path.join(self.root_dir, *biopsy_path))

        if self.biopsy_patches_only:
            return self.transforms_['biopsy'](biopsy) if self.transforms_ else biopsy

        mask_path = patch_path[:]
        mask_path.insert(-1, 'mask_patches')
        mask = Image.open(os.path.join(self.root_dir, *mask_path)) 

        patch = {'biopsy' : biopsy, 'mask' : mask} #, 'labels' : dict(labels)}
        return self.transforms_(patch) if self.transforms_ else patch
        
    def __len__(self):
        return len(self.patches_frame)

    def __repr__(self):
        return f"BarrettDataset object with number of\nDatasets: {self.N_datasets()}\
            \n\tWSIs: {self.N_wsi_names()}\n\t\tBiopsies: {self.N_biopsies()}\n\t\t\tPatches: {self.N_patches()}"

    def patch_size(self):
        return self._patch_size

    def N_datasets(self):
        return len(self.patches_frame.groupby(level='Dataset'))

    def N_wsi_names(self):
        return len(self.patches_frame.groupby(level='WSI_name'))

    def N_biopsies(self):
        return len(self.patches_frame.groupby(level='Biopsy'))

    def N_patches(self):
        return len(self)

    def N_patches_loc(self, loc_tuple):
        return len(self.patches_frame.loc[loc_tuple])


class Transforms_Dict(object):

    def __init__(self, transforms_):
        super().__init__()
        self.transforms_ = transforms_

    def __call__(self, sample):
        return {k : t(sample[k]) for k, t in self.transforms_.items()}

# %%
if __name__ == "__main__":
    from torchvision.transforms import transforms
    out_dir = 'data/Barrett20x'

    patch_size = (224, 224)
    stride = (112, 112)
    threshold = 0.33

    root_dir = f'{out_dir}_patched_ps{patch_size[0]}_{patch_size[1]}_str{stride[0]}_{stride[1]}_thr{str(threshold).replace(".", "")}'

    labels_list = []
    N_biopsies=None

    
    transforms_dict = {'biopsy' : transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]),
        'mask' : transforms.Compose([
        transforms.ToTensor(),
        ])}
    transforms_=Transforms_Dict(transforms_dict)

    barrett_data = BarrettDataset(root_dir=root_dir, labels_list=labels_list, transforms_=transforms_, N_biopsies=N_biopsies, biopsy_patches_only=False)
    print(barrett_data)
    # loader = DataLoader(barrett_data, 16, True)

    # for i, patch in enumerate(loader):
    #     print([(k, v.shape) for k, v in patch.items()])

