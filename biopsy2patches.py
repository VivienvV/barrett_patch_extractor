# %%
import os
from pathlib import Path
import itertools

import numpy as np
import matplotlib.pyplot as plt

import torch

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from utils import open_PIL_file, make_gif, plot_patch

# %%

class Biopsy2Patches():
    def __init__(self, 
                biopsy_dir,
                out_dir,
                patch_size=(224, 224),
                stride=(224,224),
                save_gif=True):

        self.biopsy_dir = biopsy_dir
        self.out_dir = out_dir
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        self.patch_size = patch_size
        self.stride = stride
        self.save_gif = save_gif

        self.biopsy = open_PIL_file(biopsy_dir, 'biopsy.png', mode='RGB', to_numpy=True)
        self.mask = open_PIL_file(biopsy_dir, 'mask.png', mode='L', to_numpy=True)

        self.N_rows_cols = self.get_N_rows_cols()
        self.N_patches = self.N_rows_cols[0] * self.N_rows_cols[1]

        self.patches_xy = self.get_xy_patches()
        self.generate_patches()

    def generate_patches(self):
        height, width = self.patch_size
        vertical_stride, horizontal_stride = self.stride

        mask_patches = torch.from_numpy(self.mask).unfold(0, height, vertical_stride).unfold(1, width, horizontal_stride)
        mask_patches = mask_patches.contiguous().view(self.N_patches, height, width)

        biopsy_patches = torch.from_numpy(self.biopsy).unfold(0, height, vertical_stride).unfold(1, width, horizontal_stride)
        biopsy_patches = biopsy_patches.contiguous().view(-1, 3, height, width).permute(0, 2, 3, 1)
        
        patches_accepted = np.zeros(self.N_patches)
        for i, (biopsy_patch, mask_patch, xy) in enumerate(zip(biopsy_patches, mask_patches, self.patches_xy)):
            # if i == 4:
            #     plot_patch(self.mask, patch_mask, xy, self.patch_size)
            #     plot_patch(self.biopsy, patch_biopsy, xy, self.patch_size)
            #     plt.show()
            patch_accepted = self.is_valid_patch(mask_patch)

            if patch_accepted:
                patches_accepted[i] = patch_accepted
                self.save_patch(biopsy_patch, mask_patch, xy)

        if self.save_gif:
            gif = make_gif('patches_from_mask.gif', self.mask, mask_patches, patches_accepted, self.patches_xy, self.patch_size)
            gif = make_gif('patches_from_biopsy.gif', self.biopsy, biopsy_patches, patches_accepted, self.patches_xy, self.patch_size)

    def is_valid_patch(self, patch):
        return True

    def get_N_rows_cols(self):
        return np.floor(((np.array(self.mask.shape) - np.array(self.patch_size)) / self.stride) + 1).astype(int)

    def get_xy_patches(self):
        N_rows, N_cols = self.N_rows_cols
        vertical_stride, horizontal_stride = self.stride
        xy = np.zeros((N_rows, N_cols, 2))
        for row, col in itertools.product(range(N_rows), range(N_cols)):
            xy[row, col] = np.array([col * horizontal_stride, row * vertical_stride])
        return xy.reshape(self.N_patches, 2)

    def save_patch(self, biopsy, mask, xy):
        
# %%


biopsy_dir = "Barrett20x/Bolero/RB0004_HE/RB0004_HE-1"
out_dir = os.path.join(biopsy_dir, "patched")

Biopsy2Patches(biopsy_dir, out_dir, patch_size=(800, 400), stride=(400, 200))
# %%
