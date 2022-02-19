# %%
import os
from pathlib import Path
import itertools

import numpy as np
import torch

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from utils.utils import open_PIL_file, save_PIL_file
from utils.plotting import make_gif

# %%

class Biopsy2Patches():
    def __init__(self, 
                root_dir,
                out_dir,
                patch_size=(224, 224),
                stride=(224,224),
                threshold=0.1,
                save_gif=False):

        self.root_dir = root_dir
        self.out_dir = out_dir

        self.biopsy_out_dir = os.path.join(*[out_dir, 'biopsy_patches'])
        Path(self.biopsy_out_dir).mkdir(parents=True, exist_ok=True)
        self.mask_out_dir = os.path.join(*[out_dir, 'mask_patches'])
        Path(self.mask_out_dir).mkdir(parents=True, exist_ok=True)

        self.patch_size = patch_size
        self.stride = stride
        self.threshold = threshold
        self.save_gif = save_gif

        self.biopsy = open_PIL_file(root_dir, 'biopsy.png', mode='RGB', to_numpy=True)
        self.mask = open_PIL_file(root_dir, 'mask.png', mode='L', to_numpy=True)

        self.N_rows_cols = self.get_N_rows_cols()
        self.N_patches = self.N_rows_cols[0] * self.N_rows_cols[1]

        self.patches_xy = self.get_xy_patches()
        self.extract_patches()

    def extract_patches(self):
        height, width = self.patch_size
        vertical_stride, horizontal_stride = self.stride

        mask_patches = torch.from_numpy(self.mask).unfold(0, height, vertical_stride).unfold(1, width, horizontal_stride)
        mask_patches = mask_patches.contiguous().view(self.N_patches, height, width)

        biopsy_patches = torch.from_numpy(self.biopsy).unfold(0, height, vertical_stride).unfold(1, width, horizontal_stride)
        biopsy_patches = biopsy_patches.contiguous().view(-1, 3, height, width).permute(0, 2, 3, 1)
        
        accepted_patches = self.is_valid_patch(mask_patches, self.threshold)
        for i, accepted in enumerate(accepted_patches):
            if accepted:
                self.save_patch(biopsy_patches[i], mask_patches[i], self.patches_xy[i])
                # self.create_csv_entry(biopsy_patches[i], mask_patches[i], self.patches_xy[i])
 
        if self.save_gif:
            print("\tCreating gif files...")
            print("\t\tFor masks...")
            make_gif(os.path.join(*[self.out_dir, 'patches_from_mask.gif']), self.mask, mask_patches, accepted_patches, self.patches_xy, self.patch_size)
            print("\t\tFor biopsy...")
            make_gif(os.path.join(*[self.out_dir, 'patches_from_biopsy.gif']), self.biopsy, biopsy_patches, accepted_patches, self.patches_xy, self.patch_size)

    def is_valid_patch(self, mask_patches, threshold):
        return torch.count_nonzero(mask_patches, dim=(1, 2)) / (self.patch_size[0] * self.patch_size[1]) > threshold

    def get_N_rows_cols(self):
        return np.floor(((np.array(self.mask.shape) - self.patch_size) / self.stride) + 1).astype(int)

    def get_xy_patches(self):
        N_rows, N_cols = self.N_rows_cols
        vertical_stride, horizontal_stride = self.stride
        xy = np.zeros((N_rows, N_cols, 2), dtype=int)
        for row, col in itertools.product(range(N_rows), range(N_cols)):
            xy[row, col] = np.array([col * horizontal_stride, row * vertical_stride], dtype=int)
        return xy.reshape(self.N_patches, 2)

    def create_csv_entry(self, biopsy, mask, xy):
        pass

    def save_patch(self, biopsy, mask, xy):
        file_name = f'x{xy[0]}_y{xy[1]}.png'
        
        save_PIL_file(biopsy, self.biopsy_out_dir, file_name, mode='RGB')
        save_PIL_file(mask, self.mask_out_dir, file_name, mode='L')
        pass

if __name__ == "__main__":
    root_dir = 'Barrett20x'
    out_dir = 'Barrett20x_Patched'
    datasets = ["Bolero", "LANS", "RBE", "RBE_Nieuw"]

    patch_size = (224, 224)
    stride = (112, 112)
    threshold = 0.1

    save_fig = False

    for dataset in datasets:
        for WSI_name in os.listdir(os.path.join(root_dir, dataset)):
            for biopsy in os.listdir(os.path.join(*[root_dir, dataset, WSI_name])):

                biopsy_root_dir = os.path.join(*[root_dir, dataset, WSI_name, biopsy])
                biopsy_out_dir = os.path.join(*[out_dir, dataset, WSI_name, biopsy])
                print(f"Extracting pathces from {biopsy_root_dir} into {biopsy_out_dir}")
                Biopsy2Patches(biopsy_root_dir, biopsy_out_dir, patch_size, stride, threshold, save_fig)
                
    # root_dir = "Barrett20x/Bolero/RB0004_HE/RB0004_HE-1"

# %%