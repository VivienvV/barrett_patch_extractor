# Barrett Esophagus Whole Slide Image (WSI) patch extractor outline
Outline of new patch extractor

## Implementation
The pipeline to create patches from whole slide images can be split up into parts. This is a draft of the pipeline and things that should/could be included.

### Step 1: WSI --> Biopsies
The first step is to extract biopsies from the tiff files and convert the xml annotations to masks. Inputs are the TIFF file with its corresponding XML file containing annotations for the N biopsies. Output for each of these files is N biopsies and masks.

class WSI2Biopsies(root_dir,
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

Returns:
biopsies (list): List of N biopsies
masks (list): List of N masks
"""

### Step 2: Biopsies → Patches
The second step is to convert all biopsies and their corresponding masks into smaller patches. 

class Biopsy2Patches(root_dir,
                     out_dir,
                     patch_size=(224, 224),
                     stride=(224,224),
                     threshold=0.15,
                     save_patches=True,
                     save_gif=False):
"""Function to extract patches from single biopsy given the parameters.

            Parameters:
            root_dir (str): Root directory of biopsy containing biopsy.png, mask.png and exclude.png
            out_dir (str): Directory to store patches in.
            patch_size (tuple): Tuple for size of extracted patches. Resulting patches will have size of (H x W).
            stride (tuple): Tuple for stride of extracting patches in both vertical and horizontal direction (V_stride, H_stride).
            threshold (float): Patches with an area of Squamous, NDBE, LGD and HGD together larger than threshold are saved.
            save_patches (bool): If True, patches of biopsy and mask are save in biopsy_patches and mask_patches.
            save_gif (bool): If True, a gif of all selected patches highlighted will be made for biopsy and mask


            Returns:
            Saves patches of biopsy and mask in out_dir in folders biopsy_patches and mask_patches respectfully. Also creates dictionaries
            for patch level and biopsy level labels.            
            """

### Step 3: Creating dataset files/splits
To be done…

## Datasets
Relevant datasets on AMC servers (/data/archief/AMC-data/Barrett/):
* ASL
   * N patients: 36
   * prefix(es): “ASLxx_K_HE” (K is 3 or 1. difference?)
* Bolero
   * N patients: 51
   * prefix(es): “RB00xx_HE”
* LANS
   * N patients: 34
   * prefix(es): “RLxxx_0x_ox_ox_HE” (last three x can be 1-5; meaning?)
* LANS-Tissue
   * N patients: 42
   * prefix(es): “RLxxx_0x_ox_ox_HE” (last three x can be 1-7; meaning?)
* RBE
   * N patients: 187
   * prefix(es): “RBE-00xxx_HE”, “RBET17-xxxxx_HE_r”, “RBET18-xxxxx_HE-r_BIG”  “ROCTxx_r-HEx” (last x is number). r is Roman Numeral
   * Note: Missing xml P53 staining for “RBE-00121_P53.tiff”, “RBE-00123_P53.tiff”, “RBE-00124_P53.tiff”
* RBE_Nieuw
   * N patients: 25
   * prefix(es): “RBE-00xxx_HE”

Datasets to exclude:
* Maastricht- OLD_DATA
   * DO NOT INCLUDE
* temp:
   * DO NOT INCLUDE: Seems to be subset of RBE

## Structure
We build our whole dataset by combining all of the above datasets. We want for our final dataset to be structured according to dataset, patients, biopsies and patches to be flexible. This also allows us to balance subsets according to labels and to split the dataset into train/validation/test independent of dataset and patients. 

* Dataset (ASL, LANS, etc...)
   * WSI_ID
      * Biopsies
         * Patches
Patch data can be stored in several ways. We could store all patches as individual png files or (preferably) combine patches into larger files such as HDF5 files for faster I/O operations.

## Labels
Labels are important for training down-stream tasks and also balancing our dataset. We want to collect individual labels per patch. Also, for future use we want to collect labels on biopsy and WSI_ID level.

### Annotation classes in tiff:
Each tiff file has a corresponding xml file with annotations. Annotations can be on a Glandular and Tissue level which is indicated by -G and -T respectively. The following annotations are used:
* Biopsy-Outlines: Outlines of biopsies defined by 4 coordinates. One WSI can contain several biopts
* Squamous(-G? and -T): Squamous epithelium. (By definition only -T?)
* NDBE(-G and -T): non-dysplastic Barrett’s esophagus tissue
* LGD(-G and -T): Low-grade dysplasia
* HGD(-G and -T): High-grade dysplasia
* E-Stroma: Tissue which lies in area between glandular structures
* Exclude: Regions of tissue which are excluded for several reasons. IMPORTANT: regions within areas annotated as Exclude can contain other annotations.


There is also some tissue which is not annotated but is (maybe) still relevant to distinguish between: 
* Background: Area which lies within Biopsy-Outlines and contains no tissue
* Unannotated stroma: Tissue which lies within Biopsy-Outlines but has no further annotation (This tissue has no annotation in the datasets because it is deemed diagnostically less relevant).
* Out of bounds (OOB) exclude: Tissue which lies outside of the Biopsy-Outlines annotations. This tissue is excluded as well
Annotation classes to Labels:
Some of these annotations can be turned directly into labels but others need to be constructed (e.g. background). Labels are stored as masks and some other special labels are constructed. 


The table below shows the definitions of labels from annotations. Background and stroma are seen as one labeled class but could also be interpreted as two different labels. 


Label_id 	Label_name     Corresp. Ann.                             Remarks
   0        Background     Biopsy-Outlines + No Tissue               Background and Stroma can be separate or combined classes. Argument for splitting: One has 
                                                                     tissue and other does not. Combining them could lead to confusion if the sampled patch contains only stroma in real world application.
                                                                     Argument against splitting: Stroma is diagnostically not really relevant. By splitting them, labels will get cluttered and training resources will be devoted to less relevant labels 
	
   1        Stroma         E-Stroma or Biopsy-Outlines + Tissue

	2        Squamous       Squamous-T
	
	3        NDBE           NDBE-G or NDBE-T                          Combine Glandular/Tissue level annotations if Tissue level enabled.
	
	4        LGD            LGD-G or LGD-T                            Combine Glandular/Tissue level annotations if Tissue level enabled.

	5     	HGD         	HGD-G or HGD-T                            Combine Glandular/Tissue level annotations into one if Tissue level enabled.
	-1       Exclude     	Exclude                                	Excluded areas do not really contain a label. A separate mask can be created which can be either applied or not on patches.

Besides creating masks used for segmentation tasks, some other special labels can be collected and stored. These labels can be used for different down-stream tasks and can be over on different levels besides patch level. Some of these labels are:
* Center pixel label of Patch
* Area (in pixels or ratio) of each label in Patch, Biopsy and WSI/Patient separately
* Single label for Patch, Biopsy and WSI/Patient separately according to highest priority label (HGD > LGD > NDBE > Squamous >= Stroma or Background)


## Magnification
TIFFs have several zoom levels stored within them. Zoom level 0 is ‘40x’, which is 0.25 mu/pixel (4 pixels per mu). Below is a table with zoom levels and their corresponding scales.


Level       	Zoom     	mu/pixels   	pixels/mu
	0        	40x         	0.25        	4        	
   1        	20x         	0.5         	2
	2        	10x         	1              1
	3        	5x             2              0.5
	etc…