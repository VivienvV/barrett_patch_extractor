import os

from PIL import Image, ImageDraw
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib import colors, patches

import imageio

# make a color map of fixed colors
LISTEDCOLORMAPS = {'blues' : ['ghostwhite', '#d4f0fc', '#89d6fb', '#02a9f7', '#02577a', '#01303f'],
                    'contrasts' : ['ghostwhite', '#0a9ad7', 'orchid', '#9ad70a', '#ffcc06', '#ff3f3f']
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

MAGN2LEVEL = {  40  : 0,
                20  : 1,
                10  : 2,
                5   : 3,
                # 2.5 : 4,
                # 1.25 : 5,
                # 0.625 : 6,
                # 0.3125 : 7,
                # 0.15625 : 8
                }

"""
=========================================================
=                                                       =
=                   MANAGING FILE                       =
=                                                       =
=========================================================
"""

def open_PIL_file(biopsy_path, file_name, mode=None, to_numpy=False):
    img = Image.open(os.path.join(biopsy_path, file_name))

    if mode is not None:
        img = img.convert(mode)
    return np.array(img) if to_numpy else img

def save_PIL_file(biopsy_path, file_name, file, mode=None):
    file = Image.fromarray(file.detach().cpu().numpy() , mode=mode)
    file.save(os.path.join(*[biopsy_path, file_name]), "PNG")
    file.close()

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
    ax3.imshow(exclude)

    ax2.set_title('Mask')
    mask_imshow = ax2.imshow(mask, interpolation='none', cmap=DEFAULT_CMAP, norm=DEFAULT_NORM)
    create_legend(mask_imshow)
    if save_fig: plt.savefig(os.path.join(*[biopsy_path, "biopsy_mask_exclude_fig.png"]), format='png')
    if show_fig: plt.show()

def make_gif(file_name, mask, patches_mask, accepted_patches, xy_list, patch_size):
    frames = []
    for patch, accepted, xy in zip(patches_mask, accepted_patches, xy_list):
        frame = plot_patch(mask, patch, xy, patch_size, accepted)
        frame.canvas.draw()
        frames.append(Image.frombytes('RGB', frame.canvas.get_width_height(), frame.canvas.tostring_rgb()))
        plt.close()
    imageio.mimsave(file_name, frames, 'GIF', fps=3)

def plot_patch(full_img, patch, xy, patch_size, accepted=False):
    h, w = patch_size
    color = 'lime' if accepted else 'orangered'
    fig, axs = plt.subplots(1, 2)

    if full_img.shape[-1] == 3:
        axs_imshow = axs[0].imshow(full_img)
        axs[1].imshow(patch)
    else:
        axs_imshow = axs[0].imshow(full_img, interpolation='none', cmap=DEFAULT_CMAP, norm=DEFAULT_NORM)
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

"""
=========================================================
=                                                       =
=                   MISCELLANEOUS                       =
=                                                       =
=========================================================
"""

def magnification2level(magnification, magnification2level=MAGN2LEVEL):
    """Function to get level from specified magnifications

    Parameters:
    magnification (int): Zoom level to use in TIFF file

    Returns:
    level (int): Level to extract in TIFF file to get correct zoom
    """
    return magnification2level[magnification]

def polygons2str(polygons, WSI_name, dataset):
    print(f"Annotation polygons found for {WSI_name} in {dataset} dataset")
    for ann_level, ann_groups in polygons.items():
        print(f"\t{ann_level} annotations in WSI:")
        for ann_group, polys in ann_groups.items():
            print(f"\t\tFor {ann_group} found {len(polys)} annotations.")
    print("")