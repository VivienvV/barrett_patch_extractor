import os

from PIL import Image
import numpy as np

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
    print(f"Annotation polygons found for {WSI_name}:")
    for ann_level, ann_groups in polygons.items():
        print(f"\t{ann_level} annotations in WSI:")
        for ann_group, polys in ann_groups.items():
            print(f"\t\tFor {ann_group} found {len(polys)} annotations.")
    print("")