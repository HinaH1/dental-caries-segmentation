"""
segmentation_data_setup.py

This script sets up the CariXray segmentation dataset for YOLO training.

The dataset from Dang et al. comes with all images and labels in flat folders,
using filename prefixes to indicate the split (train_, val_, test_). YOLO 
needs a specific folder structure to work, so this script reorganises 
everything into the correct layout and generates the data.yaml file.

It does the following:
    1. Creates the train/val/test folder structure YOLO expects
    2. Sorts images into the right folders based on their filename prefix
    3. Matches each image to its label file
    4. Creates an empty label file for any image that has no annotation,
       since YOLO requires every image to have a corresponding label file
    5. Generates data.yaml with the dataset paths and class information

Author: HinaH1
"""

import os
import shutil

# Define the path to the unzipped segmentation dataset 
base = os.path.expanduser("/usr/tmp/final_year_project/SEGMENTATION")
images_src = os.path.join(base, "images")
labels_src = os.path.join(base, "labels")

# YOLO expects separate train/val/test folders each with images and labels subfolders
for split in ['train', 'val', 'test']:
    os.makedirs(f"{base}/{split}/images", exist_ok=True)
    os.makedirs(f"{base}/{split}/labels", exist_ok=True)

# The dataset uses filename prefixes to indicate the split
# train_ and train2_ go to train, val_ to val, test_ to test
for f in os.listdir(images_src):
    if f.startswith('train_') or f.startswith('train2_'):
        shutil.copy(os.path.join(images_src, f), f"{base}/train/images/{f}")
    elif f.startswith('val_'):
        shutil.copy(os.path.join(images_src, f), f"{base}/val/images/{f}")
    elif f.startswith('test_'):
        shutil.copy(os.path.join(images_src, f), f"{base}/test/images/{f}")

# The dataset uses filename prefixes to indicate the split
# train_ and train2_ go to train, val_ to val, test_ to test
for split in ['train', 'val', 'test']:
    image_dir = f"{base}/{split}/images"
    label_dir = f"{base}/{split}/labels"
    
    for img_file in os.listdir(image_dir):
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + ".txt"
        src = os.path.join(labels_src, label_file)
        dst = os.path.join(label_dir, label_file)
        
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            open(dst, 'w').close()

# Generate the data.yaml file that YOLO needs to locate the dataset
# nc is the number of classes which is just caries in this case
yaml_content = f"""path: {base}
train: train/images
val: val/images
test: test/images
nc: 1
names: ['caries']
"""

with open(os.path.join(base, "data.yaml"), 'w') as f:
    f.write(yaml_content)

# Print a quick summary to verify the split sizes look right
for split in ['train', 'val', 'test']:
    images = len(os.listdir(f"{base}/{split}/images"))
    labels = len(os.listdir(f"{base}/{split}/labels"))
    print(f"{split}: {images} images, {labels} labels")

print("data.yaml created")
print("Setup complete")