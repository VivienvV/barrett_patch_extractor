import os
import logging

from PIL import Image
import numpy as np
import pandas as pd
import json

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

def save_PIL_file(file, biopsy_path, file_name, mode=None):
    file = Image.fromarray(file.detach().cpu().numpy(), mode=mode)
    file.save(os.path.join(*[biopsy_path, file_name]), "PNG")
    file.close()

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

def polygons2str(polygons, WSI_name):
    logging.info(f"Annotation polygons found for {WSI_name}:")
    for ann_level, ann_groups in polygons.items():
        logging.info(f"\t{ann_level} annotations in WSI:")
        for ann_group, polys in ann_groups.items():
            logging.info(f"\t\tFor {ann_group} found {len(polys)} annotations.")
    logging.info("")

def create_dataframes(root_dir, datasets):
    database_frame_biopsy_tuples = []
    database_frame_patches_tuples = []
    for dataset in datasets:
        for WSI_name in os.listdir(os.path.join(root_dir, dataset)):
            for biopsy in os.listdir(os.path.join(*[root_dir, dataset, WSI_name])):
                database_frame_biopsy_tuples.append((dataset, WSI_name, biopsy))
                database_frame_patches_tuples.append((dataset, WSI_name, biopsy, '0.png'))

    label_to_collect_biopsy = ['height', 'width', 'Background', 'Stroma', 'Squamous', 'NDBE', 'LGD', 'HGD', 'center_label', 'dominant_label', 'highest_label']
    df_biopsy = pd.DataFrame(
        np.nan,
        pd.MultiIndex.from_tuples(
            database_frame_biopsy_tuples,
            names=['Dataset', 'WSI_name', 'Biopsy']
        ),
        label_to_collect_biopsy
    )

    label_to_collect_patches = ['x', 'y', 'Background', 'Stroma', 'Squamous', 'NDBE', 'LGD', 'HGD', 'center_label', 'dominant_label', 'highest_label']
    df_patches = pd.DataFrame(
        np.nan,
        pd.MultiIndex.from_tuples(
            database_frame_patches_tuples,
            names=['Dataset', 'WSI_name', 'Biopsy', 'Patch_idx']
        ),
        label_to_collect_patches
    )
    return df_biopsy, df_patches


def save_dataframes(out_dir, df_biopsy, df_patches, meta_data, verbose):
    df_biopsy = df_biopsy.dropna(axis=0, how="any", subset=df_biopsy.columns)
    df_patches = df_patches.dropna(axis=0, how="any", subset=df_patches.columns)

    df_biopsy = df_biopsy.astype(int)
    df_biopsy.sort_index(axis=0, level=['Dataset', 'WSI_name', 'Biopsy'], ascending=True, inplace=True)
    df_biopsy.to_csv(os.path.join(*[out_dir, 'labels_biopsy_level.csv']))

    df_patches = df_patches.astype(int)
    df_patches.sort_index(axis=0, level=['Dataset', 'WSI_name', 'Biopsy', 'Patch_idx'], ascending=True, inplace=True)
    df_patches.to_csv(os.path.join(*[out_dir, 'labels_patches_level.csv']))

    with open(os.path.join(*[out_dir, 'meta_data.json']), 'w') as fp:
        json.dump(meta_data, fp)

    if verbose:
        logging.info("\nLabels at biopsy level:\n", df_biopsy)
        logging.info("\nLabels at patch level\n", df_patches)
        logging.info("\nMeta data stored: \n", meta_data)