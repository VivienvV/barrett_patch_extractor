# %%
import os
from collections import defaultdict
from torch.utils.data import Dataset

from PIL import Image
# import cv2


class BarrettDataset(Dataset):
    def __init__(self, root_dir, N_biopsies=None, transforms_=None, patches_only=True):
        super().__init__()
        
        self.root_dir = root_dir
        self.patches_only = patches_only

        self.transforms_ = transforms_

        self.biopsies = os.listdir(root_dir)[:N_biopsies]
        self.patches = defaultdict(lambda : [])
        
        self.patchIDs = []
        for biopsy in self.biopsies:
            biopsy_patches = os.path.join(biopsy, 'patches')
            self.patches[biopsy] = os.listdir(os.path.join(root_dir, biopsy_patches))
            
            [self.patchIDs.append((biopsy, patch)) 
                    for patch in self.patches[biopsy]]

        # TODO get patch size by unpacking patches/ mask
        # Get H x W from mask which is always last element
        # self._patch_size = self.__getitem__(0)[-1].shape[1:]
        self._patch_size = "NotImplemented"

    def __getitem__(self, index):
        biopsy, patch_id = self.patchIDs[index]

        patch_path = os.path.join(*[self.root_dir, biopsy, 'patches', patch_id])
        patch = self.transforms_['patches'](Image.open(patch_path).convert("RGB"))
        
        if self.patches_only:
            mask = []
        else:
            mask_path = os.path.join(*[self.root_dir, biopsy, 'masks', patch_id])
            mask = self.transforms_['masks'](Image.open(mask_path).convert("L"))

        return patch, mask

    def __len__(self):
        return len(self.patchIDs)

    def __repr__(self):
        return f"BarrettDataset object with {self.N_patches()} patches \
from {self.N_biopsies()} biopsies of size {self.patch_size()}"

    def patch_size(self):
        return self._patch_size

    def N_biopsies(self):
        return len(self.biopsies)

    def N_patches(self):
        return len(self)

    def N_patches_biopsy(self, biopsy):
        return len(self.patches[biopsy])



# %%
if __name__ == "__main__":
    # from torchvision.transforms import transforms
    
    # root_dir = '../data/patched_data/patched_20x'
    # transforms_ = {'patches' : transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #     ]),
    #     'masks' : transforms.Compose([
    #     transforms.ToTensor(),
    #     ])}

    # barrett_data = BarrettDataset(root_dir=root_dir, transforms_=transforms_)
    pass

