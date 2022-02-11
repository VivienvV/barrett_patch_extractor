# %%
import os
import warnings
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import xml.etree.ElementTree as ET
import openslide

# %%
ANNOTATION_CLASSES = ["Biopsy-Outlines", "Squamous-T", "NDBE-G", "NDBE-T",
                    "LGD-G", "LGD-T", "HGD-G", "HGD-T", "Exclude"]

MAGN2LEVEL = {  40 : 0,
                20 : 1,
                10 : 2,
                5 : 3,
                # 2.5 : 4,
                # 1.25 : 5,
                # 0.625 : 6,
                # 0.3125 : 7,
                # 0.15625 : 8
                }

class WSI2Biopsy():
    def __init__(self, 
                root_dir, 
                dataset, 
                WSI_name,
                out_dir,
                annotation_classes=ANNOTATION_CLASSES,
                magnification=20,
                extract_stroma=False, 
                mask_exclude=False,
                verbatim=True):
        """Function to extract N biopsies and masks from TIFF file with its corresponding XML file with annotations. 

        Parameters:
        files_path (str): String with path to tiff and xml files containing N biopsies for one WSI
        magnification (int): Integer at which magnification to the biopsies. Default is 20x
        extract_stroma (bool): If set to True, stroma regions will be separated from background regions and receive their own label. If set to False, stroma and background regions will be combined into one label.
        mask_exclude (bool): If set to True, regions annotated with “Exclude” will be masked in the resulting images.

        Returns:
        biopsies (list): List of N biopsies
        masks (list): List of N masks
        """

        self.out_dir = out_dir
        self.dataset = dataset
        self.WSI_name = WSI_name

        self.level = self.magnification2level(magnification)
        self.extract_stroma = extract_stroma
        self.mask_exclude = mask_exclude
        self.verbatim = verbatim

        # Create output dir with all needed subfolders
        self.biopsy_out_dir = os.path.join(*[out_dir, dataset, WSI_name])
        Path(self.biopsy_out_dir).mkdir(parents=True, exist_ok=True)

        # Open TIFF and XML files
        self.TIFF = openslide.OpenSlide(os.path.join(*[root_dir, dataset, WSI_name]) + '.tiff')
        self.XML = ET.parse(os.path.join(*[root_dir, dataset, WSI_name]) +'.xml').getroot()
        
        # Create dictionaries for biopsy boundaries and extract biopsies
        self.biopsy_boundaries = self.get_biopsy_boundaries()
        self.generate_biopsies()
        self.TIFF.close()

        # Create dictionaries for polygons (values) for each class (key)
        self.polygons = self.get_polygons(annotation_classes)
        for biopsy_path, biopsy_dict in self.biopsy_boundaries.items():
            self.generate_exclude(biopsy_path, biopsy_dict)
            self.generate_mask(biopsy_path, biopsy_dict)

        if mask_exclude:
            self._mask_exclude()

    def get_biopsy_boundaries(self):
        biopsy_boundaries = {}

        for biopsy in self.get_elements_in_group("Biopsy-Outlines"):
            biopsy_path = self.create_biopsy_dir(biopsy)
            biopsy_coords = self.get_coords(biopsy)

            top_left = np.amin(biopsy_coords, axis=0)
            bottom_right = np.amax(biopsy_coords, axis=0)
            
            # Location: (x, y) tuple giving the top left pixel in the level 0 reference frame
            location = tuple(top_left)

            # Size: (width, height) tuple giving the region size
            size = tuple(((bottom_right - top_left) / 2 ** self.level).astype(int))

            biopsy_boundaries[biopsy_path] = {'top_left' : top_left, 
                                                'location' : location,
                                                'size': size}
        return biopsy_boundaries

    def get_polygons(self, annotation_classes):
        """Structure of polygons"""

        polygons = {annotation_class : [self.get_coords(polygon_group) for polygon_group in self.get_elements_in_group(annotation_class)] 
                                                                        for annotation_class in annotation_classes}

        if self.verbatim:
            print(f"Annotation polygons found for {self.WSI_name} in {self.dataset} dataset")
            [print(f"\tFor {ann_group} found {len(polys)} annotations in WSI") for ann_group, polys in polygons.items()]
        return polygons

    def generate_biopsies(self):
        """Biopsies"""
        for biopsy_path, biopsy_dict in self.biopsy_boundaries.items():
            if self.verbatim: print(f"Extracting biopsy {biopsy_path}")
            location, size = biopsy_dict['location'], biopsy_dict['size']
            biopsy_img = self.TIFF.read_region(location, self.level, size)
            biopsy_img.save(os.path.join(*[biopsy_path, "biopsy.png"]), "PNG")


    def generate_exclude(self, biopsy_path, biopsy_dict):
        top_left, size = biopsy_dict['top_left'], biopsy_dict['size']
        polygons = self.polygons["Exclude"]

        exclude = Image.new(mode='1', size=tuple(size))
        for polygon in polygons:
            print("drawing polyon", polygon)
            # ImageDraw.Draw(exclude).polygon(polygon, fill=1, outline=1)

    def generate_mask(self):
        raise NotImplementedError
    

    def _mask_exclude(self):
        raise NotImplementedError

    # ================== HELPER FUNCTIONS ================================
    def magnification2level(self, magnification):
        return MAGN2LEVEL[magnification]

    def create_biopsy_dir(self, biopsy):
        index = biopsy.attrib["Name"].rfind(" ")
        roi_id = int(biopsy.attrib["Name"][index:])
        biopsy_path = os.path.join(*[self.biopsy_out_dir, f"{self.WSI_name}-{roi_id}"])
        Path(biopsy_path).mkdir(parents=True, exist_ok=True)
        return biopsy_path

    def get_elements_in_group(self, group):
        return self.XML.findall(f".//Annotation/.[@PartOfGroup='{group}']")

    def get_coords(self, element):
        coords = np.array([[int(re.split(',|\.', coordinates.attrib['X'])[0]), int(re.split(",|\.", coordinates.attrib['Y'])[0])]
                                    for coordinates in element.iter('Coordinate')])

        if len(coords) < 3:
            warnings.warn("Warning: Found polygon group with less than 3 elements")
        return coords

# %%
if __name__ == "__main__":
    out_dir = "Barrett20x"
    root_dir = "TIFFs"
    # dataset = "RBE"
    # WSI_name = "ROCT24_IX-HE3"
    # dataset = "RBE"
    # WSI_name = "RBET18-02665_HE-I_BIG"
    # dataset = "ASL"
    # WSI_name = "ASL28_1_HE"

    datasets = ["ASL", "Bolero", "LANS", "LANS-Tissue", "RBE", "RBE_Nieuw"]

    for dataset in datasets:
        WSI_names = [file.split(".")[0] for file in os.listdir(os.path.join(root_dir, dataset)) if file.endswith(".tiff")]
        print("DATASET ", dataset)
        for WSI_name in WSI_names:
            foobar = WSI2Biopsy(root_dir, dataset, WSI_name, out_dir, magnification=10)


    # %%
    for biopsy_path, biopsy_dict in foobar.biopsy_boundaries.items():
        top_left, bottom_right = biopsy_dict['top_left'], biopsy_dict['bottom_right']
        polygons = foobar.polygons["Exclude"]
        size = bottom_right - top_left

        exclude = Image.new(mode='1', size=tuple(size))
        print(exclude.size)
        for polygon in polygons:

            print(polygon - top_left)
            ImageDraw.Draw(exclude).polygon(polygon - top_left, outline=1)
        
        
        exclude.save('foobar_exclude.png')
        # exclude.show()
    
    # %%

    # import xml.dom.minidom

    # dom = xml.dom.minidom.parse(os.path.join(*[root_dir, dataset, WSI_name]) +'.xml') # or xml.dom.minidom.parseString(xml_string)
    # pretty_xml_as_string = dom.toprettyxml()
    # print(pretty_xml_as_string)
    pass
# %%
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import flood_fill
from copy import copy

size = [100000, 100000]

exclude = Image.new(mode='1', size=tuple(size))
print('create image')

exclude_np = np.array(exclude).copy()
print('read as array')
# plt.imshow(exclude_np)
# plt.show()

# polygons = np.array([[[100, 1000], [500, 4000], [250, 6000]]])
polygons = np.array([[[100, 100], [500, 400], [250, 600]]])
top_left = np.array([50, 50])
for polygon in polygons:
    print(polygon)
    polygon_tuples = [tuple(coord) for coord in polygon - top_left]
    print(polygon_tuples)
    ImageDraw.Draw(exclude).polygon(polygon_tuples, fill=1, outline=1)

# foobar_floodfill = flood_fill(exclude, )
exclude = exclude.resize((500, 500), resample=Image.NEAREST)
exclude.save('foobar_exclude.png')

exclude_np = np.array(exclude)
print(np.amax(exclude_np), np.amin(exclude_np))
print(exclude_np)
plt.imshow(exclude_np, cmap='gray')
plt.show()
# exclude.show()
# %%

# https://docs.python.org/3/library/xml.etree.elementtree.html 
# https://openslide.org/api/python/

# %%
