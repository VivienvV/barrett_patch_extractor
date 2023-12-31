from fileinput import filename
import os
from pathlib import Path
from utils.utils import open_PIL_file

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors, patches
from PIL import Image

import imageio
from tqdm import tqdm

# make a color map of fixed colors
LISTEDCOLORMAPS = {'blues' : ['ghostwhite', '#d4f0fc', '#89d6fb', '#02a9f7', '#02577a', '#01303f'],
                    'contrasts' : ['ghostwhite', '#0a9ad7', 'orchid', '#9ad70a', '#ffcc06', '#ff3f3f'],
                    'contrasts_dark_bg' : ['darkgray', '#0a9ad7', 'orchid', '#9ad70a', '#ffcc06', '#ff3f3f']
                }

DEFAULT_CMAP = colors.ListedColormap(LISTEDCOLORMAPS['contrasts'])
DEFAULT_BOUNDS= [0, 1, 2, 3, 4, 5, 6]
DEFAULT_NORM = colors.BoundaryNorm(DEFAULT_BOUNDS, DEFAULT_CMAP.N)

LABEL2ANN = {0  : "Background",
            1   : "Stroma",
            2   : "Squamous",
            3   : "NDBE",
            4   : "LGD",
            5   : "HGD"
            }

ANN2LABEL = {"Background" : 0,
            "Stroma" : 1,
            "Squamous" : 2,
            "NDBE" : 3,
            "LGD" : 4,
            "HGD" : 5
            }

"""
=========================================================
=                                                       =
=                   PLOTTING                            =
=                                                       =
=========================================================
"""

def plot_biopsy(biopsy_path, save_fig=False, show_fig=True):
    biopsy = open_PIL_file(biopsy_path, 'biopsy.png')
    exclude = open_PIL_file(biopsy_path, 'exclude.png')
    mask = open_PIL_file(biopsy_path, 'mask.png')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 16))
    ax1.set_title('Biopsy')
    ax1.imshow(biopsy)

    ax3.set_title('Exclude')
    ax3.imshow(exclude, interpolation='none')

    ax2.set_title('Mask')
    mask_imshow = ax2.imshow(mask, interpolation='none', cmap=DEFAULT_CMAP, norm=DEFAULT_NORM)
    create_legend(mask_imshow)
    if save_fig: plt.savefig(os.path.join(*[biopsy_path, "biopsy_mask_exclude_fig.png"]), format='png')
    plt.show() if show_fig else plt.close()

def make_gif(file_name, img_full, img_patches, accepted_patches, xy_list, patch_size, only_accepted=False):
    with imageio.get_writer(file_name, 'GIF', fps=3) as writer:
        for patch, accepted, xy in tqdm(zip(img_patches, accepted_patches, xy_list), total=len(xy_list)):
            if only_accepted and not accepted:
                continue
            frame = plot_patch(img_full, patch, xy, patch_size, accepted)
            frame.canvas.draw()
            frame = Image.frombytes('RGB', frame.canvas.get_width_height(), frame.canvas.tostring_rgb())
            writer.append_data(np.array(frame))
            plt.close()
            frame.close()
        writer.close()

def plot_patch(img_full, patch, xy, patch_size, accepted=False):
    h, w = patch_size
    color = 'lime' if accepted else 'orangered'
    fig, axs = plt.subplots(1, 2)

    if img_full.shape[-1] == 3:
        axs_imshow = axs[0].imshow(img_full)
        axs[1].imshow(patch)
    else:
        axs_imshow = axs[0].imshow(img_full, interpolation='none', cmap=DEFAULT_CMAP, norm=DEFAULT_NORM)
        axs[1].imshow(patch, interpolation='none', cmap=DEFAULT_CMAP, norm=DEFAULT_NORM)
        create_legend(axs_imshow)
    
    # Create a Rectangle patch
    rect = patches.Rectangle(xy, w, h, linewidth=1, edgecolor=color, facecolor=color, alpha=0.2)
    # Add the patch to the Axes
    axs[0].add_patch(rect)
    return fig

def create_legend(imshow):
    # From: https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib

    values = np.arange(max(LABEL2ANN.keys()) + 1)
    # get the colors of the values, according to the 
    # colormap used by imshow
    colors = [ imshow.cmap(imshow.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    handles = [ patches.Patch(color=colors[i], label=LABEL2ANN[value]) for i, value in enumerate(values) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    # return patches

def save_color_mask_file(mask, color_mask_out_dir, file_name, show=False):
    fig, ax = plt.subplots()
    ax.imshow(mask, interpolation='none', cmap=DEFAULT_CMAP, norm=DEFAULT_NORM)
    ax = plt.gca()
    plt.axis('off')
    plt.savefig(os.path.join(*[color_mask_out_dir, file_name]), format='png', bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    else:
        plt.close()



"""
=========================================================
=                                                       =
=                   DATASET PROBING                     =
=                                                       =
=========================================================
"""

def dataset_probing(data_frame_path):
    save_dir = os.path.join(data_frame_path, 'data_probing')
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    data_frame = pd.read_csv(os.path.join(data_frame_path, 'labels_patches_level.csv'), index_col=['Dataset', 'WSI_name', 'Biopsy', 'Patch_idx'])
    norm_data_frame = data_frame.groupby('Dataset')[['Background', 'Stroma', 'Squamous', 'NDBE', 'LGD', 'HGD']].apply(lambda x: x / (224 * 224))
    mean_dataset = norm_data_frame.groupby('Dataset').mean()
    std_dataset = norm_data_frame.groupby('Dataset').std()

    mean_dataset.plot(kind='bar', cmap=colors.ListedColormap(LISTEDCOLORMAPS['contrasts_dark_bg']))
    plt.savefig(os.path.join(save_dir, 'mean_dataset.png'))
    plt.close()

    std_dataset.plot(kind='bar', cmap=colors.ListedColormap(LISTEDCOLORMAPS['contrasts_dark_bg']))
    plt.savefig(os.path.join(save_dir, 'std_dataset.png'))
    plt.close()

    for label in ['center_label', 'dominant_label', 'highest_label']:
        foobar = data_frame.groupby('Dataset')[label].value_counts().unstack(fill_value=int(0))
        foobar.plot(kind='bar', cmap=colors.ListedColormap(LISTEDCOLORMAPS['contrasts_dark_bg']), title=label)
        plt.savefig(os.path.join(save_dir, f'{label}_dataset.png'))
        plt.close()

