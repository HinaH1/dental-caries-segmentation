"""
detection_data_setup.py

This script sets up the CariXray detection dataset for YOLO training.

It works in the same way as segmentation_data_setup.py but points to the 
detection folder of the CariXray dataset. The detection labels use YOLO 
bounding box format rather than polygon coordinates:
    class_id x_center y_center width height
where all values are normalised between 0 and 1.

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

# Points to the detection folder within the CariXray dataset
base = os.path.expanduser("/usr/tmp/final_year_project/DETECTION/DETECTION")
images_src = os.path.join(base, "images")
labels_src = os.path.join(base, "labels")

# YOLO needs separate train/val/test folders each with images and labels subfolders
for split in ['train', 'val', 'test']:
    os.makedirs(f"{base}/{split}/images", exist_ok=True)
    os.makedirs(f"{base}/{split}/labels", exist_ok=True)

# Sort images into the correct split folder based on filename prefix
# The CariXray dataset uses train_, train2_, val_ and test_ as prefixes
for f in os.listdir(images_src):
    if f.startswith('train_') or f.startswith('train2_'):
        shutil.copy(os.path.join(images_src, f), f"{base}/train/images/{f}")
    elif f.startswith('val_'):
        shutil.copy(os.path.join(images_src, f), f"{base}/val/images/{f}")
    elif f.startswith('test_'):
        shutil.copy(os.path.join(images_src, f), f"{base}/test/images/{f}")

# Match each image to its label file
# If no label exists, create an empty file to keep the structure consistent
# YOLO requires every image to have a corresponding label file
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

# Generate the data.yaml file YOLO needs to find the dataset
# nc is the number of classes — just caries in this case
yaml_content = f"""path: {base}
train: train/images
val: val/images
test: test/images
nc: 1
names: ['caries']
"""

with open(os.path.join(base, "data.yaml"), 'w') as f:
    f.write(yaml_content)

# Generate the data.yaml file YOLO needs to find the dataset
# nc is the number of classes — just caries in this case
for split in ['train', 'val', 'test']:
    images = len(os.listdir(f"{base}/{split}/images"))
    labels = len(os.listdir(f"{base}/{split}/labels"))
    print(f"{split}: {images} images, {labels} labels")

print("data.yaml created")
print("Setup complete")