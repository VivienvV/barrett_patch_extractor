# %%
import os
from pathlib import Path
import itertools
import argparse

import numpy as np
import torch

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from utils.utils import open_PIL_file, save_PIL_file, create_dataframes, save_dataframes
from utils.plotting import make_gif

DATASETS = ["Bolero", "LANS", "RBE", "RBE_Nieuw", "ASL"] #, "LANS-Tissue"]
# %%
class Biopsy2Patches():
    def __init__(self, 
                 root_dir,
                 out_dir,
                 patch_size=(224, 224),
                 stride=(224,224),
                 threshold=0.15,
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

        self.label_dict_biopsy = self.create_dict_entry(torch.from_numpy(self.mask), self.mask.shape, self.mask.shape)
        self.label_dict_patches = {}
        self.patch_idx = 0

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
                self.label_dict_patches[self.patch_idx] = self.create_dict_entry(mask_patches[i], self.patches_xy[i], self.patch_size)
                self.patch_idx += 1
 
        if self.save_gif:
            print("\tCreating gif files...")
            print("\t\tFor masks...")
            make_gif(os.path.join(*[self.out_dir, 'patches_from_mask.gif']), self.mask, mask_patches, accepted_patches, self.patches_xy, self.patch_size)
            print("\t\tFor biopsy...")
            make_gif(os.path.join(*[self.out_dir, 'patches_from_biopsy.gif']), self.biopsy, biopsy_patches, accepted_patches, self.patches_xy, self.patch_size)

    def is_valid_patch(self, mask_patches, threshold):
        print(mask_patches.shape)
        ones = torch.sum((mask_patches == 1), dim=(1, 2))
        print(ones.shape, ones)
        print(torch.count_nonzero(mask_patches, dim=(1, 2)))
        nonzero_excl_one = torch.count_nonzero(mask_patches, dim=(1, 2)) - ones

        print(nonzero_excl_one.shape, nonzero_excl_one)
        return nonzero_excl_one / (self.patch_size[0] * self.patch_size[1]) > threshold

    def get_N_rows_cols(self):
        return np.floor(((np.array(self.mask.shape) - self.patch_size) / self.stride) + 1).astype(int)

    def get_xy_patches(self):
        N_rows, N_cols = self.N_rows_cols
        vertical_stride, horizontal_stride = self.stride
        xy = np.zeros((N_rows, N_cols, 2), dtype=int)
        for row, col in itertools.product(range(N_rows), range(N_cols)):
            xy[row, col] = np.array([col * horizontal_stride, row * vertical_stride], dtype=int)
        return xy.reshape(self.N_patches, 2)

    def create_dict_entry(self, mask, xy, patch_size):
        area_in_pixels = torch.bincount(mask.flatten(), minlength=6)
        center_pixel = mask[int(patch_size[0] / 2), int(patch_size[1] / 2)].item()

        patch_dict = {  'x': xy[0],
                        'y': xy[1],
                        'Background': area_in_pixels[0].item(),
                        'Stroma': area_in_pixels[1].item(),
                        'Squamous': area_in_pixels[2].item(),
                        'NDBE' : area_in_pixels[3].item(),
                        'LGD' : area_in_pixels[4].item(),
                        'HGD' : area_in_pixels[5].item(),
                        'center_label': center_pixel,
                        'dominant_label' : area_in_pixels.argmax().item(),
                        'highest_label' : torch.nonzero(area_in_pixels).squeeze().max().item()
                    }

        return patch_dict

    def save_patch(self, biopsy, mask, xy):
        file_name = f'x{xy[0]}_y{xy[1]}.png'
        
        save_PIL_file(biopsy, self.biopsy_out_dir, file_name, mode='RGB')
        save_PIL_file(mask, self.mask_out_dir, file_name, mode='L')

def extract_patches(config):
    stride = tuple((int(config.stride[0]), int(config.stride[1])))
    patch_size = tuple((int(config.patch_size[0]), int(config.patch_size[1])))

    if config.verbose:
        print(f"Extracting patches from {config.root_dir} as root directory and saving in {config.out_dir}")
        print(f"Datasets: {config.datasets}")
        print(f"Patch size: {patch_size}")
        print(f"Stride: {stride}")
        print(f"Threshold: {config.threshold}")

        print(f"Extracting GIFs of extracted masks and biopsy: {config.save_gif}\n")

    df_biopsy, df_patches = create_dataframes(config.root_dir, config.datasets)

    for dataset in config.datasets:
        if config.verbose: print(f"=================== EXTRACTING DATASET {dataset} ===================")
        for WSI_name in os.listdir(os.path.join(config.root_dir, dataset)):
            for biopsy in os.listdir(os.path.join(*[config.root_dir, dataset, WSI_name])):
                if config.verbose: print(f"\tExtracting patches from {biopsy} of WSI {WSI_name}...")
                biopsy_root_dir = os.path.join(*[config.root_dir, dataset, WSI_name, biopsy])
                biopsy_out_dir = os.path.join(*[config.out_dir, dataset, WSI_name, biopsy])

                B2P = Biopsy2Patches(biopsy_root_dir, biopsy_out_dir, patch_size, stride, config.threshold, config.save_gif)
                if config.verbose: print(f'\t\tExtracted {max(B2P.label_dict_patches.keys())} patches of the {len(B2P.patches_xy)}')

                df_biopsy.loc[(dataset, WSI_name, biopsy), :] = list(B2P.label_dict_biopsy.values())
                for idx, patch_dict in B2P.label_dict_patches.items():
                    df_patches.loc[(dataset, WSI_name, biopsy, idx), :] = list(patch_dict.values())

    save_dataframes(config.out_dir, df_biopsy, df_patches, config.verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration of parameters for Biopsy2Patches processing.")

    # Directories
    parser.add_argument('--root_dir', type=str, default='data/Barrett20x',
                        help='Path to dataset with Biopsies')
    parser.add_argument('--out_dir', type=str, default='data/Barrett20x_patched',
                        help='Path to store patches of dataset')

    # Dataset specifications
    parser.add_argument('--datasets', nargs='+', default=DATASETS,
                        help='Datasets to extract')
    parser.add_argument('--patch_size', nargs='+', default=(224, 224),
                        help='Tuple of desired patch size with dimensions HxW')
    parser.add_argument('--stride', nargs='+', default=(112, 112),
                        help='Tuple of desired stride for vertical and horizontal direction respectively')
    parser.add_argument('--threshold', type=float, default=0.15,
                        help='Threshold for determining if patch is valid or not')

    # Extra flags
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='If True, print statements during parsing')
    parser.add_argument('--save_gif', action='store_true', default=False,
                        help='Create and save gif for every biopsy and mask')

    config = parser.parse_args()

    extract_patches(config)
