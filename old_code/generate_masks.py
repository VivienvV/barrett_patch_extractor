# %%
import os
import numpy as np
from PIL import Image, ImageDraw, ImageChops
import xml.etree.ElementTree as ET
import openslide
Image.MAX_IMAGE_PIXELS = None
from scipy import ndimage as ndi
from skimage import exposure
# import concurrconda install scikit-imageent.futures

image_path = "./TIFFs/Bolero/"
export_path = "./Barrett20x/old_code/"

# which level to extract
level = 1

# list of images for mask extraction

lijst = []
l = os.listdir(image_path)
for file in l:
    if file.endswith("tiff"):
        index = file.find(".")
        lijst.append(file[:index])

lijst.sort()

# lijst = ['ASL21_1_HE']
print(len(lijst))


for img in lijst:
# def create_masks(img):
    #     print(os.path.join(image_path, img + ".tiff" ))
    image = openslide.OpenSlide(os.path.join(image_path, img + ".tiff"))
    xml = os.path.join(image_path, img + ".xml")

    width1, height1 = image.level_dimensions[0]
    print(width1, height1)

    annot = ET.parse(xml)
    annot_root = annot.getroot()

    # determine outline coordinates of the total ROI

# ========================== BIOPSY OUTLINES ============================================
    counter = 0
    for l1 in annot_root.iter('Annotation'):
        if l1.attrib["PartOfGroup"] == "Biopsy-Outlines":
            index = l1.attrib["Name"].rfind(" ")
            roi_id = int(l1.attrib["Name"][index:])
            counter += 1
            ltx = 0
            lty = 0
            rbx = 0
            rby = 0
            for l2 in l1.iter('Coordinates'):
                for l3 in l2.iter('Coordinate'):
                    x = int(float(l3.attrib["X"].split(",")[0]))
                    y = int(float(l3.attrib["Y"].split(",")[0]))
                    if ltx == 0:
                        ltx = x
                        lty = y
                        rbx = x
                        rby = y
                    if x < ltx:
                        ltx = x
                    if y < lty:
                        lty = y
                    if x > rbx:
                        rbx = x
                    if y > rby:
                        rby = y

            # save the ROI of the H&E
            # First, check if export folder exists, if not, create it
            #             print(ltx,lty,rbx,rby)
            expfolder = img + "-" + str(roi_id)
            if not os.path.exists(os.path.join(export_path, expfolder)):
                os.makedirs(os.path.join(export_path, expfolder))

            HE = image.read_region((ltx, lty), level, (int((rbx - ltx) / 2 ** level), int((rby - lty) / 2 ** level)))
            HE.save(os.path.join(export_path, expfolder, "img" + ".png"), "PNG")

            width = HE.size[0]
            height = HE.size[1]
            print(width, height)
            
            # Cut same areas out of the WSI-mask
            box = (int(ltx), int(lty), int(rbx), int(rby))
            print("Box: ", box)

# ========================== TISSUE DETECTION ============================================

            # Detect tissue, and create a ouline mask
            # This mask will be combined with annotation-masks, to remove artifacts, and clean border annotations
            
            im = np.asarray(HE)
            HE.close()
            im_green = im[:, :, 1]
            p2, p98 = np.percentile(im_green, (2, 98))
            outline_unfiltered = exposure.rescale_intensity(im_green, in_range=(p2, p98))
            outline_unfiltered = outline_unfiltered < 220
            outline_unfiltered = Image.fromarray(np.uint8(outline_unfiltered * 255))
            #             outline_filtered = ndi.binary_fill_holes(outline_unfiltered)
            label_objects, nb_labels = ndi.label(outline_unfiltered)
            sizes = np.bincount(label_objects.ravel())
            mask_sizes = sizes > 10000
            #             print("1:", len(mask_sizes))
            mask_sizes[0] = 0
            outline_filtered = mask_sizes[label_objects]
            outline1 = Image.fromarray(np.uint8(outline_filtered * 255)).convert('1')
            # outline1.save(os.path.join(export_path, expfolder,'outline1.png'))
            
            #             outline_copy.save(os.path.join(export_path, expfolder,'outline_raw.png'))
            # find holes with size treshold and fill these
            outline2 = ImageChops.invert(outline1)
            label_objects, nb_labels = ndi.label(outline2)
            sizes = np.bincount(label_objects.ravel())
            mask_sizes = sizes < 10000
            #             print("2:", len(mask_sizes))
            mask_sizes[0] = 0
            outline_filtered2 = mask_sizes[label_objects]
            outline2 = Image.fromarray(np.uint8(outline_filtered2 * 255)).convert('1')
            # outline2.save(os.path.join(export_path, expfolder,'outline2.png'))
            
            outline3 = ImageChops.logical_or(outline1, outline2)
            outline1.close()
            outline2.close()
            outline3.save(os.path.join(export_path, expfolder, 'outline3.png'))

# ========================== E-Stroma ============================================
            # mask which contains stroma elements within glandular structures.
            # Also this mask must be substracted from the glandular masks
            annotationList = []
            for l1 in annot_root.iter('Annotation'):
                if l1.attrib["PartOfGroup"] == "E-Stroma":
                    polygon = []
                    for l2 in l1.iter('Coordinates'):
                        if len(l2) > 3:
                            point = []
                            for l3 in l2.iter('Coordinate'):
                                x = int(float(l3.attrib["X"].split(",")[0]))
                                y = int(float(l3.attrib["Y"].split(",")[0]))
                                point.append((x, y))
                            polygon.append(point)
                    annotationList.append(polygon)
            #             print("Mask1: ", annotationList)

            print("making stroma thing")
            estroma = Image.new('1', (int(width1), int(height1)), 0)
            print("drawing stroma thing")
            for a in annotationList:
                coordinates = []
                for b in a:
                    ImageDraw.Draw(estroma).polygon(b, outline=1, fill=1)
            print("cropping")
            estroma = estroma.crop(box)
            print("resize")
            estroma = estroma.resize((width, height), Image.ANTIALIAS).convert('1')
            print('saving')
            estroma.save(os.path.join(export_path, expfolder, "estroma.png"), "PNG")
            estroma = np.asarray(estroma)  # convert image to array, for subtraction operation later...
            
            # create all binary masks at level 0; From these masks submasks will be cropped

# ========================== Squamous-T ============================================
            # mask for squamous epithelium
            annotationList = []
            for l1 in annot_root.iter('Annotation'):
                if l1.attrib["PartOfGroup"] == "Squamous-T":
                    polygon = []
                    for l2 in l1.iter('Coordinates'):
                        if len(l2) > 3:
                            point = []
                            for l3 in l2.iter('Coordinate'):
                                x = int(float(l3.attrib["X"].split(",")[0]))
                                y = int(float(l3.attrib["Y"].split(",")[0]))
                                point.append((x, y))
                            polygon.append(point)
                    annotationList.append(polygon)
            #             print("Mask1: ", annotationList)
            mask1 = Image.new('1', (int(width1), int(height1)), 0)
            for a in annotationList:
                coordinates = []
                for b in a:
                    ImageDraw.Draw(mask1).polygon(b, outline=1, fill=1)
            mask1 = mask1.crop(box)
            mask1 = mask1.resize((width, height), Image.ANTIALIAS).convert('1')
            mask1 = ImageChops.logical_and(mask1, outline3)
            mask1.save(os.path.join(export_path, expfolder, "mask1.png"), "PNG")
            mask1.close()

# ========================== NDBE-G ============================================
            # mask for NDBE-G
            annotationList = []
            for l1 in annot_root.iter('Annotation'):
                if l1.attrib["PartOfGroup"] == "NDBE-G":
                    polygon = []
                    for l2 in l1.iter('Coordinates'):
                        if len(l2) > 3:
                            point = []
                            for l3 in l2.iter('Coordinate'):
                                x = int(float(l3.attrib["X"].split(",")[0]))
                                y = int(float(l3.attrib["Y"].split(",")[0]))
                                point.append((x, y))
                            polygon.append(point)
                    annotationList.append(polygon)
            #             print("Mask3: ", annotationList)
            mask3 = Image.new('1', (int(width1), int(height1)), 0)
            for a in annotationList:
                for b in a:
                    ImageDraw.Draw(mask3).polygon(b, outline=1, fill=1)
            mask3 = mask3.crop(box)
            mask3 = mask3.resize((width, height), Image.ANTIALIAS).convert('1')
            mask3 = ImageChops.logical_and(mask3, outline3)
            mask3 = Image.fromarray(np.logical_and(np.asarray(mask3), np.invert(estroma)))
            mask3.save(os.path.join(export_path, expfolder, "mask3.png"), "PNG")
            mask3.close()

# ========================== LGD-G ============================================
            #             # mask for LGD-G
            annotationList = []
            for l1 in annot_root.iter('Annotation'):
                if l1.attrib["PartOfGroup"] == "LGD-G":
                    polygon = []
                    for l2 in l1.iter('Coordinates'):
                        if len(l2) > 3:
                            point = []
                            for l3 in l2.iter('Coordinate'):
                                x = int(float(l3.attrib["X"].split(",")[0]))
                                y = int(float(l3.attrib["Y"].split(",")[0]))
                                point.append((x, y))
                            polygon.append(point)
                    annotationList.append(polygon)
            #             print("Mask5: ", annotationList)
            mask5 = Image.new('1', (int(width1), int(height1)), 0)
            for a in annotationList:
                for b in a:
                    ImageDraw.Draw(mask5).polygon(b, outline=1, fill=1)
            mask5 = mask5.crop(box)
            mask5 = mask5.resize((width, height), Image.ANTIALIAS).convert('1')
            mask5 = ImageChops.logical_and(mask5, outline3)
            mask5 = Image.fromarray(np.logical_and(np.asarray(mask5), np.invert(estroma)))
            mask5.save(os.path.join(export_path, expfolder, "mask5.png"), "PNG")
            mask5.close()

# ========================== HGD-G ============================================
            #             # 4. mask for HGD-G
            annotationList = []
            for l1 in annot_root.iter('Annotation'):
                if l1.attrib["PartOfGroup"] == "HGD-G":
                    polygon = []
                    for l2 in l1.iter('Coordinates'):
                        if len(l2) > 3:
                            point = []
                            for l3 in l2.iter('Coordinate'):
                                x = int(float(l3.attrib["X"].split(",")[0]))
                                y = int(float(l3.attrib["Y"].split(",")[0]))
                                point.append((x, y))
                            polygon.append(point)
                    annotationList.append(polygon)
            # |
            mask7 = Image.new('1', (int(width1), int(height1)), 0)
            for a in annotationList:
                for b in a:
                    ImageDraw.Draw(mask7).polygon(b, outline=1, fill=1)
            mask7 = mask7.crop(box)
            mask7 = mask7.resize((width, height), Image.ANTIALIAS).convert('1')
            mask7 = ImageChops.logical_and(mask7, outline3)
            mask7 = Image.fromarray(np.logical_and(np.asarray(mask7), np.invert(estroma)))
            mask7.save(os.path.join(export_path, expfolder, "mask7.png"), "PNG")
            mask7.close()

# ========================== Exclude ============================================
            # 5. "mask" with excluded areas
            annotationList = []
            for l1 in annot_root.iter('Annotation'):
                if l1.attrib["PartOfGroup"] == "Exclude":
                    polygon = []
                    for l2 in l1.iter('Coordinates'):
                        if len(l2) > 3:
                            point = []
                            for l3 in l2.iter('Coordinate'):
                                x = int(float(l3.attrib["X"].split(",")[0]))
                                y = int(float(l3.attrib["Y"].split(",")[0]))
                                point.append((x, y))
                            polygon.append(point)
                    annotationList.append(polygon)
            #     print("Exclude: ", annotationList)
            exclude = Image.new('1', (int(width1), int(height1)), 0)
            for a in annotationList:
                for b in a:
                    ImageDraw.Draw(exclude).polygon(b, outline=1, fill=1)
            exclude = exclude.crop(box)
            exclude = exclude.resize((width, height), Image.ANTIALIAS).convert('1')
            #             exclude = ImageChops.logical_and(mask1, outline3)
            exclude.save(os.path.join(export_path, expfolder, "exclude.png"), "PNG")
            exclude.close()
            

# with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
#     executor.map(create_masks, lijst)
# %%
