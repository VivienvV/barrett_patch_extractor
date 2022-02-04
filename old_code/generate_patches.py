import os
import os.path
from PIL import Image
import operator
import numpy as np
import csv
import concurrent.futures

"""
Generate patches from a 8-bit multiclass png
First, select which classes should be included in the patch. 
All non selected should be assigned to background class (0).
Next, remaining classes must be reassigned a class ID

CHECK IF ALL RELEVANT CLASSES HAVE BEEN REASSIGNED THE CORRECT CLASS ID 
"""

Image.MAX_IMAGE_PIXELS = None

image_path = "/mnt/data/barrett/masks"
patches_folder = "/mnt/data/barrett/4classtraining/data/"

classes_count = 5 # number of classes in total (including background class!)

size = 512  # tile size (squared)
step = 128  # stride
area = size * size
max_count = 50


# counters for storing patch-totals
counter_list = []

# initialize counters for each class with zer
for i in range(classes_count):
    counter_list.append(0)

# initialize file list for images to extract patches
lijst = []
folders = os.listdir(image_path)
for folder in folders:
    lijst.append(folder)
lijst.sort()

# lijst = ['ASL01_3_HE-2']

print(lijst)

def reclassify_mask(mask):
    # reassign pixel values to combine classes
    mask[mask == 1] = 1  # squamous
    mask[mask == 2] = 2  # NDBE
    mask[mask == 3] = 3  # LGD
    mask[mask == 4] = 4  # HGD
    return(mask)


def create_patches(file, thr=0.1):
    image = np.array(Image.open(os.path.join(image_path, file, "img.png")).convert('RGB'))
    mask = np.array(Image.open(os.path.join(image_path, file, "mch.png")))
    # mask = reclassify_mask(mask)

    # check if export folder (with name of filename) exists, if not, create
    if not os.path.isdir(os.path.join(patches_folder, file)):
        os.mkdir(os.path.join(os.path.join(patches_folder, file)))

    width = image.shape[0]
    height = image.shape[1]

    counter = 0
    temp_counter_list = list()
    class_flags = list()
    for c in range(classes_count):
        temp_counter_list.append(0)
        class_flags.append(False)
    patch_id = 0
    x = 0
    while x < (width - size):
        y = 0
        while y < (height - size):
            flag = True  # keep track if current patch is already extracted
            print(f'patch:{patch_id}:  ({x}, {y})')
            np_mask = np.copy(mask[x:x + size, y:y + size])
            # print(f'shape: {np_mask_img.shape} size: {np_mask_img.size} som: {np_mask_img.sum()}')
            # inspect tile
            for c in range(classes_count):
                if (np_mask == c + 1).sum() / area > thr:
                    class_flags[c] = True  # there is normal epithelium present in this patch
                else:
                    class_flags[c] = False

                print(f'\t {c + 1}: {(np_mask == c + 1).sum()};\t n{c + 1}: {class_flags[c]}')

            print(f'\t Flag: {flag}')
            # ignore the last class since it is always recognized as irrelevant background!!
            for c in range(classes_count - 1):
                # check if normal tissue is present, not more than max patches of this type are extracted
                if class_flags[c] and temp_counter_list[c] < max_count and flag:
                    flag = False  # prevent extraction of this patch below
                    counter += 1
                    mask_img = Image.fromarray(np_mask)
                    mask_img.save(
                        os.path.join(patches_folder, file, file + "-c" + str(c + 1) + "-" + str(counter) + "-mask.png"),
                        "PNG")
                    np_image = image[x:x + size, y:y + size]
                    img = Image.fromarray(np_image)
                    img.save(os.path.join(patches_folder, file, file + "-c" + str(c + 1) + "-" + str(counter) + ".png"),
                             "PNG")
                    print(f' image and mask saved (L{c + 1}) {flag} id: {counter}')
                    del np_mask
            # ignore the last class since it is always recognized as irrelevant background!!
            # count number of occurrences of a specific class in a patch
            for c in range(classes_count - 1):
                if class_flags[c] and not flag:
                    temp_counter_list[c] += 1

            # reset flags
            for c in range(classes_count):
                class_flags[c] = False

            y += step
            patch_id += 1
        x += step

    # insert csv.writer code for logfile
    logdata = [file]
    print(f' Filename: {file}\n'
          f'total patches: {counter}\n')
    for c in range(classes_count - 1):
        print(f'class {c + 1} patches: {temp_counter_list[c]}\n')
        logdata.append(temp_counter_list[c])
    print('\n')
    # summarize results
    for c in range(classes_count - 1):
        counter_list[c] += temp_counter_list[c]

    print(f' TOTALS:')
    for c in range(classes_count - 1):
        print(f'class {c + 1} patches: {counter_list[c]}\n')
    print('\n')
    writer.writerow(logdata)
    return


if __name__ == "__main__":

    # with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
    #     executor.map(create_patches, lijst)
    logfile = open(os.path.join(patches_folder, "logfile.csv"), 'w')
    writer = csv.writer(logfile)
    header = ["filename", "squamous(n)", "ndbe(n)", "lgd(n)", "hgd(n)"]
    writer.writerow(header)

    for file in lijst:
        create_patches(file)

    logfile.close()