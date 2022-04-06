# %%
import os
import logging
import re
from pathlib import Path
import argparse

import numpy as np

from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None # Change max pixel setting to allow to open big images

from scipy import ndimage as ndi
from skimage import exposure

import xml.etree.ElementTree as ET
import openslide

from utils.utils import magnification2level, polygons2str
from utils.plotting import plot_biopsy

# %%
# DEFAULT MAPPINGS
DATASETS = ["ASL", "Bolero", "LANS", "RBE"]#, "LANS-Tissue"]

ANNOTATION_CLASSES_DICT = { "Special"  : ["Biopsy-Outlines", "E-Stroma", "Exclude"], 
                            "G-level"  : ["NDBE-G", "LGD-G", "HGD-G"],
                            "T-level"  : ["Squamous-T"]
                            }

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

# %%
class WSI2Biopsy():
    def __init__(self, 
                 root_dir, 
                 out_dir,
                 WSI_name,
                 annotation_classes_dict=ANNOTATION_CLASSES_DICT,
                 label_map=ANN2LABEL,
                 magnification=20,
                 extract_stroma=False, 
                 mask_exclude=False,
                 verbose=True):
        """Function to extract N biopsies and masks from TIFF file with its corresponding XML file with annotations. 

        Parameters:
        root_dir (str): String with path to root directory containing WSI directories.
        out_dir (str): String with path to save biopsies and masks.
        WSI_name (str): String with path to folder of tiff and xml files containing N biopsies for one WSI.
        annotation_classes_dict (dict): Dictionary of annotation groups based level of annotation to extract in XML files.
        label_map (dict): Dictionary with label corresponding to annotation group.
        magnification (int): Integer at which magnification to the biopsies. Options are: [40, 20, 10, 5, ...].
        extract_stroma (bool): If set to True, stroma regions will be separated from background regions and be combined with E-Stroma class. If set to False, stroma and background regions will be combined into one label.
        mask_exclude (bool): If set to True, regions annotated with Exclude will be masked in the resulting images.
        verbose (bool): If set to True, prints with progress updates will be made.
        """

        self.annotation_classes_dict = annotation_classes_dict
        self.label_map = label_map

        self.level = magnification2level(magnification)
        self.extract_stroma = extract_stroma
        self.mask_exclude = mask_exclude
        self.verbose = verbose

        # Create output dir with all needed subfolders
        self.out_dir = out_dir
        self.WSI_name = WSI_name
        Path(os.path.join(*[out_dir, WSI_name])).mkdir(parents=True, exist_ok=True)

        # Open TIFF and XML files
        self.TIFF = openslide.OpenSlide(os.path.join(*[root_dir, WSI_name]) + '.tiff')
        self.XML = ET.parse(os.path.join(*[root_dir, WSI_name]) +'.xml').getroot()
        
        # Create dictionaries for polygons (values) for each class (key)
        self.polygons = self.get_polygons(annotation_classes_dict)            

        # Create dictionaries for biopsy boundaries and extract biopsies
        self.biopsy_dict = self.get_biopsy_dict()

        # Create for every biopsy a biopsy.png, exclude.png and mask.png
        for biopsy_path in self.biopsy_dict.keys():
            if self.verbose: logging.info(f"Extracting biopsy {biopsy_path}")

            self.extract_biopsy(biopsy_path)
            self.extract_mask(biopsy_path)
            self.extract_exclude(biopsy_path)

            if mask_exclude:
                self._mask_exclude(biopsy_path)

            if self.verbose: logging.info('')

    """
    =================================================================================
    =                                                                               =
    =                           EXTRACTION OF PNGs                                  =
    =                                                                               =
    =================================================================================
    """

    def extract_biopsy(self, biopsy_path):
        """Function to extract and save one biopsy from self.biopsy_dict at certain zoom level. Each biopsy
        is saved as biopsy.png in its subfolder of WSI_name.
        """
        if self.verbose: logging.info("\tGenerating biopsy...")
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

    def extract_mask(self, biopsy_path):
        """Function to create and save 8bit mask for annotation regions. Each mask
        is saved as mask.png in its subfolder of WSI_name.

        Parameters:
        biopsy_path (str): Path to subfolder of WSI_name corresponding to biopsy.
        label_map (dict): Dictionary which maps each annotation class to a label.
        """
        if self.verbose: logging.info("\tExtracting mask...")
        binary_masks = {ann_level : {} for ann_level in self.annotation_classes_dict.keys()}
        # E-Stroma mask is created to substract regions inbetween glandular structures.
        binary_masks['Special']['E-Stroma'] = self.create_binary_mask(biopsy_path, 'Special', 'E-Stroma')
        # Store inverted mask for effeciency.
        e_stroma_inv = np.invert(binary_masks['Special']['E-Stroma'])

        # Create binary masks of all Glandular level structures
        for ann_class in self.annotation_classes_dict['G-level']:
            ann_class_mask = self.create_binary_mask(biopsy_path, 'G-level', ann_class)
            binary_masks['G-level'][ann_class] = np.logical_and(ann_class_mask, e_stroma_inv)

        # Update inverted mask with all Glandular structures and E-Stroma
        g_level_inv = np.invert(np.any(np.array(np.array(list(binary_masks['G-level'].values()) + [binary_masks['Special']['E-Stroma']])), axis=0))
        
        # Create binary masks of all Tissue level structures and subtract all G-level masks
        for ann_class in self.annotation_classes_dict['T-level']:
            ann_class_mask = self.create_binary_mask(biopsy_path, 'T-level', ann_class)
            binary_masks['T-level'][ann_class] = np.logical_and(ann_class_mask, g_level_inv)

        # All unnanotated regions on tissue mask are seen as stroma
        if self.extract_stroma:
            all_level_inv = np.invert(np.any(np.array(np.array(list(binary_masks['G-level'].values()) + list(binary_masks['T-level'].values()) + [binary_masks['Special']['E-Stroma']])), axis=0))
            
            unannotated_stroma = np.logical_and(all_level_inv, self.biopsy_dict[biopsy_path]['tissue_mask'])
            binary_masks['Special']['E-Stroma'] = np.logical_or(binary_masks['Special']['E-Stroma'], unannotated_stroma)

        final_mask = self.combine_binary_masks(biopsy_path, binary_masks)
        final_mask = Image.fromarray(final_mask, mode="L")
        final_mask.save(os.path.join(*[biopsy_path, "mask.png"]), "PNG")
        final_mask.close()

    def extract_exclude(self, biopsy_path):
        """Function to create and save binary mask for Exclude regions. Each mask
        is saved as exclude.png in its subfolder of WSI_name.
        """
        if self.verbose: logging.info("\tExtracting exclude...")
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

        if self.verbose: logging.info(f"\t\tNumber of pixels with conflicting labels: {len(np.argwhere(np.count_nonzero(final_mask, axis=-1) > 2))}")

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

    def create_biopsy_dir(self, biopsy):
        """Function to create subfolder for a biopsy in TIFF file using XML element

        Parameters:
        biopsy (element): XML element of biopsy annotation.

        Returns:
        biopsy_path (str): Path to subfolder for biopsy 
        """
        index = biopsy.attrib["Name"].rfind(" ")
        roi_id = int(biopsy.attrib["Name"][index:])
        biopsy_path = os.path.join(*[self.out_dir, self.WSI_name, f"{self.WSI_name}-{roi_id}"])
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
            logging.warning("Warning: Found polygon group with less than 3 elements")
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
            polygons[ann_level] = {annotation_class : [self.get_coords(polygon_group) for polygon_group in self.get_elements_in_group(annotation_class) if len(self.get_coords(polygon_group)) > 2] 
                                                                        for annotation_class in annotation_classes}
        if self.verbose: polygons2str(polygons, self.WSI_name)
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

def extract_biopsies(config):
    if config.verbose: 
        logging.info(f"Extracting datasets at {config.magnification}x magnification from {config.root_dir} as root directory and saving in {config.out_dir}")
        logging.info(f"Datasets: {config.datasets}")
        logging.info(f"Extract Stroma: {config.extract_stroma}")
        logging.info(f"Masking Exclude: {config.mask_exclude}")
        logging.info(f"Showing extracted biopsy: {config.save_fig}")
        logging.info(f"Annotation_classes_dict: {config.annotation_classes_dict}\n")

    for dataset in config.datasets:
        dataset_root_dir = os.path.join(config.root_dir, dataset)
        dataset_out_dir = os.path.join(config.out_dir, dataset)

        processed_WSI_names = os.listdir(dataset_out_dir)
        
        WSI_names = [file.split(".")[0] for file in os.listdir(dataset_root_dir) if file.endswith(".tiff")]
        if config.verbose: logging.info(f"=================== EXTRACTING DATASET {dataset} ===================\n")

        all_dataset_files = os.listdir(dataset_root_dir)

        for WSI_name in WSI_names:
            if WSI_name + '.tiff' not in all_dataset_files or WSI_name + '.xml' not in all_dataset_files:
                if config.verbose: logging.warning(f"FOUND NO TIFF OR XML FILE FOR {WSI_name}")
                continue
            if config.skip_processed and WSI_name in processed_WSI_names:
                if config.verbose: logging.info(f"Skipping {WSI_name} because it already exists in {dataset_out_dir}")
                continue
            biopsy = WSI2Biopsy(dataset_root_dir,
                            dataset_out_dir, 
                            WSI_name, 
                            annotation_classes_dict=config.annotation_classes_dict, 
                            magnification=config.magnification, 
                            extract_stroma=config.extract_stroma, 
                            verbose=config.verbose)

            if config.save_fig:
                for biopsy_path in biopsy.biopsy_dict.keys():
                    plot_biopsy(biopsy_path, save_fig=True, show_fig=False)
    return True

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration of parameters for WSI2Biopsy processing.")

    # Directories
    parser.add_argument('--root_dir', type=str, default='TIFFs',
                        help='Path to dataset with WSI')
    parser.add_argument('--out_dir', type=str, default='data/Barrett20x',
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
    parser.add_argument('--skip_processed', action='store_true', default=False,
                        help='If True, do not parse already parsed WSIs again.')
    parser.add_argument('--save_fig', action='store_true', default=False,
                        help='Save biopsy and mask side by side after extraction')

    config = parser.parse_args()
    if config.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler("logs/wsi2biopsy.log"),
                logging.StreamHandler()
            ]
        )

    finished = extract_biopsies(config)
    if finished:
        logging.info(f"\nSuccesfully extracted all datasets into {config.out_dir}")