# %%
import os
import warnings
import re
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None # Change max pixel setting to allow to open big images

from scipy import ndimage as ndi
from skimage import exposure

import xml.etree.ElementTree as ET
import openslide

# %%
DATASETS = ["ASL", "Bolero", "LANS", "RBE", "RBE_Nieuw", "LANS-Tissue"]

ANNOTATION_CLASSES_DICT = {"Special"  : ["Biopsy-Outlines", "E-Stroma", "Exclude"], 
                            "G-level" : ["NDBE-G", "LGD-G", "HGD-G"],
                            "T-level" : ["Squamous-T"]}

ANNOTATION_CLASSES_DICT_T_LEVEL = {"Special"  : ["Biopsy-Outlines", "E-Stroma", "Exclude"], 
                            "G-level" : ["NDBE-G", "LGD-G", "HGD-G"],
                            "T-level" : ["Squamous-T", "NDBE-T", "LGD-T", "HGD-T"]}

ANN2LABEL = {"Biopsy-Outlines"  : 0,
            "E-Stroma"          : 1,
            "Squamous-T"        : 2, 
            "NDBE-G"            : 3, 
            "NDBE-T"            : 3,
            "LGD-G"             : 4, 
            "LGD-T"             : 4, 
            "HGD-G"             : 5, 
            "HGD-T"             : 5
            }

LABEL2ANN = {0  : "Background",
            1   : "E-Stroma",
            2   : "Squamous-T",
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

# %%
class WSI2Biopsy():
    def __init__(self, 
                root_dir, 
                dataset, 
                WSI_name,
                out_dir,
                annotation_classes_dict=ANNOTATION_CLASSES_DICT,
                label_map=ANN2LABEL,
                magnification=20,
                extract_stroma=False, 
                mask_exclude=False,
                verbose=True,
                save_fig=False):
        """Function to extract N biopsies and masks from TIFF file with its corresponding XML file with annotations. 

        Parameters:
        root_dir (str): String with path to root directory of datasets.
        dataset (str): String with dataset name to extract files from.
        WSI_name (str): String with path to tiff and xml files containing N biopsies for one WSI.
        out_dir (str): String with path to save biopsies and masks.
        annotation_classes_dict (dict): Dictionary of annotation groups based level of annotation to extract in XML files.
        label_map (dict): Dictionary with label corresponding to annotation group.
        magnification (int): Integer at which magnification to the biopsies. Options are: [40, 20, 10, 5, ...].
        extract_stroma (bool): If set to True, stroma regions will be separated from background regions and be combined with E-Stroma class. If set to False, stroma and background regions will be combined into one label.
        mask_exclude (bool): If set to True, regions annotated with Exclude will be masked in the resulting images.
        verbose (bool): If set to True, prints with progress updates will be made.
        """

        self.out_dir = out_dir
        self.dataset = dataset
        self.WSI_name = WSI_name

        self.annotation_classes_dict = annotation_classes_dict
        self.label_map = label_map

        self.level = self.magnification2level(magnification)
        self.extract_stroma = extract_stroma
        self.mask_exclude = mask_exclude
        self.verbose = verbose
        self.save_fig = save_fig

        # Create output dir with all needed subfolders
        self.biopsy_out_dir = os.path.join(*[out_dir, dataset, WSI_name])
        Path(self.biopsy_out_dir).mkdir(parents=True, exist_ok=True)

        # Open TIFF and XML files
        self.TIFF = openslide.OpenSlide(os.path.join(*[root_dir, dataset, WSI_name]) + '.tiff')
        self.XML = ET.parse(os.path.join(*[root_dir, dataset, WSI_name]) +'.xml').getroot()
        
        # Create dictionaries for polygons (values) for each class (key)
        self.polygons = self.get_polygons(annotation_classes_dict)
        if self.verbose:
            print(f"Annotation polygons found for {self.WSI_name} in {self.dataset} dataset")
            for ann_level, ann_groups in self.polygons.items():
                print(f"\t{ann_level} annotations in WSI:")
                for ann_group, polys in ann_groups.items():
                    print(f"\t\tFor {ann_group} found {len(polys)} annotations.")
            print("")

        # Create dictionaries for biopsy boundaries and extract biopsies
        self.biopsy_dict = self.get_biopsy_dict()

        # Create for every biopsy a biopsy.png, exclude.png and mask.png
        for biopsy_path in self.biopsy_dict.keys():
            if self.verbose: print(f"Extracting biopsy {biopsy_path}\n\tGenerating biopsy...")
            self.generate_biopsy(biopsy_path)
            if self.verbose: print("\tGenerating mask...")
            self.generate_mask(biopsy_path)

            if self.verbose: print("\tGenerating exclude...")
            self.generate_exclude(biopsy_path)

            if mask_exclude:
                self._mask_exclude(biopsy_path)

            if self.save_fig: 
                if self.verbose:
                    print("\tSaving example figure...")
                self.plot_biopsy(biopsy_path)
            if self.verbose: print()

    def get_biopsy_dict(self):
        """Function to create folders for biopsy. Also collects biopsy boundaries from XML file and stores them
        in a dictionary

        Returns:
        biopsy_dict (dict): Dictionary with biopsy_path as key and dictionary as value with top_left, bottom_right
        location (in level 0) and sizes (calculated for self.level and on level 0) for every biopsy
        """
        biopsy_dict = {}

        for biopsy in self.get_elements_in_group("Biopsy-Outlines"):
            biopsy_path = self.create_biopsy_dir(biopsy)
            biopsy_coords = self.get_coords(biopsy)

            # Location = top_left: (x, y) tuple giving the top left pixel in the level 0 reference frame
            top_left = np.amin(biopsy_coords, axis=0)
            bottom_right = np.amax(biopsy_coords, axis=0)

            # Size: (width, height) tuple giving the region size
            size_out = tuple(((bottom_right - top_left) / 2 ** self.level).astype(int))
            size_l0 = tuple(bottom_right - top_left)

            biopsy_dict[biopsy_path] = {'top_left' : top_left,
                                        'bottom_right' : bottom_right, 
                                        'size_out' : size_out,
                                        'size_l0' : size_l0}
        return biopsy_dict

    """
    =================================================================================
    =                                                                               =
    =                           GENERATION OF PNGs                                  =
    =                                                                               =
    =================================================================================
    """

    def generate_biopsy(self, biopsy_path):
        """Function to extract and save one biopsy from self.biopsy_dict at certain zoom level. Each biopsy
        is saved as biopsy.png in its subfolder of WSI_name.
        """
        
        location, size_out = tuple(self.biopsy_dict[biopsy_path]['top_left']), self.biopsy_dict[biopsy_path]['size_out']

        biopsy_img = self.TIFF.read_region(location, self.level, size_out)
        biopsy_img.save(os.path.join(*[biopsy_path, "biopsy.png"]), "PNG")

        biopsy_mono = np.asarray(biopsy_img.getchannel(1))
        biopsy_img.close()
        self.biopsy_dict[biopsy_path]['tissue_mask'] = self.generate_tissue_mask(biopsy_mono)

    def generate_tissue_mask(self, biopsy_mono_channel, q=(2, 98)):
        """Function to extract mask with all tissue in biopsy.

        Parameters:
        biopsy_mono_channel (array): G-channel array of size (W x H x 1) of biopsy to create mask from.
        q (tuple): q value with low and high for np.percentile function.

        Returns:
        tissue_mask (array): Boolean mask of size (W x H) corresponding to all tissue in biopsy
        """
        # TODO Use Mathematical morphology for detecting of tissue masks: https://en.wikipedia.org/wiki/Mathematical_morphology
        # Find all tisue and filter small objects from mask
        p2, p98 = np.percentile(biopsy_mono_channel, q)
        tissue_w_holes = exposure.rescale_intensity(biopsy_mono_channel, in_range=(p2, p98)) < 220
        label_objects, _ = ndi.label(tissue_w_holes)
        sizes = np.bincount(label_objects.ravel())
        mask_sizes = sizes > 10000
        mask_sizes[0] = 0
        tissue_w_holes = mask_sizes[label_objects]
        
        # Find holes using inverse and filter out large holes
        holes = np.invert(tissue_w_holes)
        label_objects, _ = ndi.label(holes)
        sizes = np.bincount(label_objects.ravel())
        mask_sizes = sizes < 10000
        mask_sizes[0] = 0
        holes = mask_sizes[label_objects]
        
        return np.logical_or(tissue_w_holes, holes)

    def generate_mask(self, biopsy_path):
        """Function to create and save 8bit mask for annotation regions. Each mask
        is saved as mask.png in its subfolder of WSI_name.

        Parameters:
        biopsy_path (str): Path to subfolder of WSI_name corresponding to biopsy.
        label_map (dict): Dictionary which maps each annotation class to a label.
        """
        binary_masks = {ann_level : {} for ann_level in self.annotation_classes_dict.keys()}
        # E-Stroma mask is created to substract regions inbetween glandular structures.
        binary_masks['Special']['E-Stroma'] = self.create_binary_mask(biopsy_path, 'Special', 'E-Stroma')
        # Store inverted mask for effeciency.
        e_stroma_inv = np.invert(binary_masks['Special']['E-Stroma'])

        # Create binary masks of all Glandular level structures
        for ann_class in self.annotation_classes_dict['G-level']:
            ann_class_mask = self.create_binary_mask(biopsy_path, 'G-level', ann_class)
            binary_masks['G-level'][ann_class] = np.logical_and(ann_class_mask, e_stroma_inv)

        if self.verbose and self.save_fig:
            print('\t\tGlandular level...')
            g_level_mask = self.combine_binary_masks(biopsy_path, binary_masks)
            g_level_mask = Image.fromarray(g_level_mask, mode="L")
            g_level_mask.save(os.path.join(*[biopsy_path, "g_level_mask.png"]), "PNG")
            g_level_mask.close()

        # Update inverted mask with all Glandular structures and E-Stroma
        g_level_inv = np.invert(np.any(np.array(np.array(list(binary_masks['G-level'].values()) + [binary_masks['Special']['E-Stroma']])), axis=0))
        
        # Create binary masks of all Tissue level structures and subtract all G-level masks
        for ann_class in self.annotation_classes_dict['T-level']:
            ann_class_mask = self.create_binary_mask(biopsy_path, 'T-level', ann_class)
            binary_masks['T-level'][ann_class] = np.logical_and(ann_class_mask, g_level_inv)

        if self.extract_stroma:
            # Update inverted mask with all found structures
            all_level_inv = np.invert(np.any(np.array(np.array(list(binary_masks['G-level'].values()) + list(binary_masks['T-level'].values()) + [binary_masks['Special']['E-Stroma']])), axis=0))
            
            unannotated_stroma = np.logical_and(all_level_inv, self.biopsy_dict[biopsy_path]['tissue_mask'])
            binary_masks['Special']['E-Stroma'] = np.logical_or(binary_masks['Special']['E-Stroma'], unannotated_stroma)

        if self.verbose:
            print('\t\tTissue level...')
        final_mask = self.combine_binary_masks(biopsy_path, binary_masks)
        final_mask = Image.fromarray(final_mask, mode="L")
        final_mask.save(os.path.join(*[biopsy_path, "mask.png"]), "PNG")
        final_mask.close()

    def generate_exclude(self, biopsy_path):
        """Function to create and save binary mask for Exclude regions. Each mask
        is saved as exclude.png in its subfolder of WSI_name.
        """
        exclude = self.create_binary_mask(biopsy_path, "Special", "Exclude")
        exclude = Image.fromarray(exclude)
        exclude.save(os.path.join(*[biopsy_path, "exclude.png"]), "PNG")
        exclude.close()

    """
    =================================================================================
    =                                                                               =
    =                           BINARY MASK CREATION                                =
    =                                                                               =
    =================================================================================
    """

    def create_binary_mask(self, biopsy_path, ann_level, ann_class):
        """Function to create binary mask for certain annotation class. Tissue mask is applied for 
        better boundaries. Polygons found in XML file are used.
        
        Parameters:
        biopsy_path (str): Path to subfolder of WSI_name corresponding to biopsy.
        ann_level (str): Annotation level of annotation class (G-level, T-level or Special).
        ann_class (str): Annotation class to create binary mask for.

        Returns:
        mask (array): Binary mask with filled in polygons according to annotation class.
        """
        top_left, bottom_right = self.biopsy_dict[biopsy_path]['top_left'], self.biopsy_dict[biopsy_path]['bottom_right']
        size_l0 = self.biopsy_dict[biopsy_path]['size_l0']

        mask = Image.new(mode='1', size=tuple(size_l0))
        polygons = self.polygons[ann_level][ann_class]
        for polygon in polygons:
            if self.polygon_in_boundaries(polygon, top_left, bottom_right):
                norm_polygon = [tuple(p - top_left) for p in polygon]
                ImageDraw.Draw(mask).polygon(norm_polygon, fill=1, outline=1)

        size_out = self.biopsy_dict[biopsy_path]['size_out']
        mask = mask.resize(tuple(size_out), Image.ANTIALIAS)
        return np.logical_and(np.array(mask), self.biopsy_dict[biopsy_path]['tissue_mask'])

    def combine_binary_masks(self, biopsy_path, binary_masks):
        """Function that combines all binary mask into one uint8 array        
        """
        max_label = max(self.label_map.values()) + 1    # add one because of background class
        width, height = self.biopsy_dict[biopsy_path]['size_out']

        final_mask = np.zeros((height, width, max_label), dtype=bool)

        for _, ann_level_dict in binary_masks.items():
            for ann_class, bin_mask in ann_level_dict.items():
                label = self.label_map[ann_class]
                final_mask[:, :, label] = np.logical_or(final_mask[:, :, label], bin_mask)

        # Add ones everywhere in background class to ensure label is given
        final_mask[:, :, 0] = np.ones_like(final_mask[:, :, 0])

        if self.verbose:
            print(f"\t\t\tNumber of pixels with conflicting labels: {len(np.argwhere(np.count_nonzero(final_mask, axis=-1) > 2))}")

        # Return max_label minus reversed argmax to ensure highest label has priority
        return (max_label - 1 - np.argmax(final_mask[:, :, ::-1], axis=-1)).astype(np.uint8)

    def _mask_exclude(self, biopsy_path):
        """Function to mask out regions of biopsy.png and mask.png using corresponding exclude.png

        """
        raise NotImplementedError

    """
    =================================================================================
    =                                                                               =
    =                           HELPER FUNCTIONS                                    =
    =                                                                               =
    =================================================================================
    """
    def magnification2level(self, magnification):
        """Function to get level from specified magnifications

        Parameters:
        magnification (int): Zoom level to use in TIFF file

        Returns:
        level (int): Level to extract in TIFF file to get correct zoom
        """
        return MAGN2LEVEL[magnification]

    def create_biopsy_dir(self, biopsy):
        """Function to create subfolder for a biopsy in TIFF file using XML element

        Parameters:
        biopsy (element): XML element of biopsy annotation.

        Returns:
        biopsy_path (str): Path to subfolder for biopsy 
        """
        index = biopsy.attrib["Name"].rfind(" ")
        roi_id = int(biopsy.attrib["Name"][index:])
        biopsy_path = os.path.join(*[self.biopsy_out_dir, f"{self.WSI_name}-{roi_id}"])
        Path(biopsy_path).mkdir(parents=True, exist_ok=True)
        return biopsy_path

    def get_elements_in_group(self, group):
        """Function to get all elements of certain group of annotations in the XML tree.

        Parameters:
        group (str): String of PartOfGroup group to extract all elements

        Returns:
        All "Annotation" elements that are part of specified group
        """
        return self.XML.findall(f".//Annotation/.[@PartOfGroup='{group}']")

    def load_biopsy(self, biopsy_path):
        return Image.open(os.path.join(*[biopsy_path, "biopsy.png"]))

    def load_mask(self, biopsy_path, ann_level='tissue'):
        if ann_level == 'tissue':
            mask_filename = 'mask.png'
        elif ann_level == 'glandular':
            mask_filename = 'g_level_mask.png'

        return Image.open(os.path.join(*[biopsy_path, mask_filename]))

    def load_exclude(self, biopsy_path):
        return Image.open(os.path.join(*[biopsy_path, "exclude.png"]))

    def plot_biopsy(self, biopsy_path):
        biopsy = self.load_biopsy(biopsy_path)
        g_level_mask = self.load_mask(biopsy_path, 'glandular')
        mask = self.load_mask(biopsy_path, 'tissue')

        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 16))
        ax1.set_title('Biopsy')
        ax1.imshow(biopsy)

        ax2.set_title('Glandular-level Mask')
        ax2.imshow(g_level_mask, interpolation='none')

        ax3.set_title('Tissue-level Mask')
        mask_imshow = ax3.imshow(mask, interpolation='none')

        # From: https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib
        values = np.unique(mask)
        # get the colors of the values, according to the 
        # colormap used by imshow
        colors = [ mask_imshow.cmap(mask_imshow.norm(value)) for value in values]
        # create a patch (proxy artist) for every color 
        patches = [ mpatches.Patch(color=colors[i], label=LABEL2ANN[value]) for i, value in enumerate(values) ]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        plt.savefig(os.path.join(*[biopsy_path, "example_plot.png"]), format='png')
        plt.close()

    """
    =================================================================================
    =                                                                               =
    =                           POLYGON EXTRACTION                                  =
    =                                                                               =
    =================================================================================
    """

    def get_coords(self, element):
        """Function to get coordinates of polygon for certain element.

        Parameters:
        element (element): Annotation element with coordinates for polygon inside.

        Returns:
        coords (array): Array with polygon coordinates for element array[[int(x1), int(y1)], ... [int(xN), int(yN)]]
        """
        coords = np.array([[int(re.split(',|\.', coordinates.attrib['X'])[0]), int(re.split(",|\.", coordinates.attrib['Y'])[0])]
                                    for coordinates in element.iter('Coordinate')])

        if len(coords) < 3:
            warnings.warn("Warning: Found polygon group with less than 3 elements")
        return coords
    
    def get_polygons(self, annotation_classes_dict):
        """Function to collect all polygons from annotations in XML file
        
        Parameters:
        annotation_classes (list): List of strings of annotation classes in the XML file

        Returns:
        polygons (dict): Dictionary with list of polygons for each annotation class with
        structure: {ann_level : {annotation_class : polygons_list[polygon1[[x1, y1], [x2, y2]...], ..., polygonN[[x1, y1], [x2, y2]...]]}}
        """
        polygons = {}

        for ann_level, annotation_classes in annotation_classes_dict.items():
            polygons[ann_level] = {annotation_class : [self.get_coords(polygon_group) for polygon_group in self.get_elements_in_group(annotation_class)] 
                                                                        for annotation_class in annotation_classes}
        return polygons
        
    def polygon_in_boundaries(self, polygon, top_left, bottom_right):
        """Function to determine if ANY of the xy coords from polygon lie in boundary of top_left and bottom_right

        Parameters:
        polygon (array): Array with coordinates of structure [[x1, y1], [x2, y2]...]
        top_left (array): Array with x and y coordinates of top left of boundary
        bottom_right (array): Array with x and y coordinates of bottom right of boundary

        Returns:
        True if ANY xy coordinate pair lies in boundary, False else
        """
        # Any x, y coordinate of Polygon between top_left and bottom_right
        return np.amax(np.all(np.logical_and(np.greater_equal(polygon, top_left), np.less_equal(polygon, bottom_right)), axis=1))

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Directories
    parser.add_argument('--root_dir', type=str, default='TIFFs',
                        help='Path to dataset with WSI')
    parser.add_argument('--out_dir', type=str, default='Barrett20x',
                        help='Path to store biopsies of dataset')

    # Dataset specifications
    parser.add_argument('--datasets', nargs='+', default=DATASETS,
                        help='Datasets to extract')
    parser.add_argument('--annotation_classes_dict', type=dict, default=ANNOTATION_CLASSES_DICT,
                        help='Nested annotation classes dict')
    parser.add_argument('--label_map', type=dict, default=ANN2LABEL,
                        help='Dictionary that maps from labels to annotation class')
    parser.add_argument('--magnification', type=int, default=20,
                        help='Magnification to extract biopsies from WSI')

    # Extra flags
    parser.add_argument('--extract_stroma', action='store_true', default=False,
                        help='If True, assign all unannotated tissue to stroma class')                    
    parser.add_argument('--mask_exclude', action='store_true', default=False,
                        help='If True, black out Exclude regions in biopsy.png and mask.png')    
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='If True, print statements during parsing')
    parser.add_argument('--save_fig', action='store_true', default=False,
                        help='If True, save figure of Biopsy, G and T level mask')    

    config = parser.parse_args()
    print(config.datasets)

    if config.verbose: print(f"Extracting datasets at {config.magnification}x magnification from {config.root_dir} as root directory and saving in {config.out_dir}")

    for dataset in config.datasets:
        WSI_names = [file.split(".")[0] for file in os.listdir(os.path.join(config.root_dir, dataset)) if file.endswith(".tiff")]
        if config.verbose: print(f"========= EXTRACTING DATASET {dataset} ============")
        for WSI_name in WSI_names:
            _ = WSI2Biopsy(config.root_dir, 
                            dataset, 
                            WSI_name, 
                            config.out_dir, 
                            annotation_classes_dict=config.annotation_classes_dict, 
                            magnification=config.magnification, 
                            extract_stroma=config.extract_stroma, 
                            verbose=config.verbose, 
                            save_fig=config.save_fig)

# %%
# out_dir = "../Barrett20x"
# root_dir = "TIFFs"

# magnification = 20
# extract_stroma = False

# verbose = True
# save_fig = True

# annotation_classes_dict=ANNOTATION_CLASSES_DICT
# print(annotation_classes_dict)
# datasets = ["RBE_Nieuw"] #, "ASL", "LANS-Tissue"]

# if verbose:
#     print(f"Extracting datasets at {magnification}x magnification from {root_dir} as root directory and saving in {out_dir}")

# for dataset in datasets:
#     WSI_names = [file.split(".")[0] for file in os.listdir(os.path.join(root_dir, dataset)) if file.endswith(".tiff")]
#     print(f"========= EXTRACTING DATASET {dataset} ============")
#     for WSI_name in WSI_names:
#         _ = WSI2Biopsy(root_dir, dataset, WSI_name, out_dir, annotation_classes_dict=annotation_classes_dict, magnification=20, extract_stroma=extract_stroma, verbose=verbose, save_fig=save_fig)

# print("Using T-level annotations")
# out_dir = "../Barrett20x_T_level"
# annotation_classes_dict=ANNOTATION_CLASSES_DICT_T_LEVEL
# datasets = ["ASL", "Bolero", "LANS", "RBE", "RBE_Nieuw", "LANS-Tissue"]
# print(annotation_classes_dict)
# if verbose:
#     print(f"Extracting datasets at {magnification}x magnification from {root_dir} as root directory and saving in {out_dir}")

# for dataset in datasets:
#     WSI_names = [file.split(".")[0] for file in os.listdir(os.path.join(root_dir, dataset)) if file.endswith(".tiff")]
#     print(f"========= EXTRACTING DATASET {dataset} ============")
#     for WSI_name in WSI_names:
#         if WSI_name == "RBET18-02665_HE-I_BIG":
#             continue
#         _ = WSI2Biopsy(root_dir, dataset, WSI_name, out_dir, annotation_classes_dict=annotation_classes_dict, magnification=20, extract_stroma=extract_stroma, verbose=verbose, save_fig=save_fig)
