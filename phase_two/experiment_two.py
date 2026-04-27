"""
experiment_two.py

This script runs Experiment Two of Phase Two — YOLOv8-seg instance 
segmentation on the CariXray dataset.

This experiment builds directly on Experiment One by moving from bounding 
box detection to pixel-level segmentation. Rather than just identifying 
where a lesion is, the segmentation model traces the exact boundary of 
each carious lesion, which provides considerably more clinical information 
about lesion shape and extent.

A different seed to Experiment One is used as each experiment is treated 
as an independent training run. The results are evaluated on the same 
held-out test set to allow direct comparison with Experiment One and 
against the Dang et al. benchmark.

Author: HinaH1
"""

import random
import numpy as np
import torch
from ultralytics import YOLO

# Set a different seed to Experiment One as this is an independent training run
SEED = 123 

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Print environment info to confirm the GPU is available before training starts
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")

# Load the YOLOv8 nano segmentation model
model = YOLO("yolov8n-seg.pt")

print("YOLOv8 segmentation model loaded successfully")

# Train on the CariXray segmentation dataset
results = model.train(
    data="/usr/tmp/final_year_project/SEGMENTATION/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    seed=SEED,
    device=0,  # GPU
    project="/usr/tmp/final_year_project/EXPERIMENT2",
    name="yolov8n_caries_seg",
    exist_ok=True,
)

print("Training complete")

# Load the best weights saved during training and evaluate on the test set
# The test set was kept completely separate from training and validation
model = YOLO('/usr/tmp/final_year_project/EXPERIMENT2/yolov8n_caries_seg/weights/best.pt')

metrics = model.val(
    data='/usr/tmp/final_year_project/SEGMENTATION/SEGMENTATION/data.yaml',
    split='test'
)

# Print the test set results
print("\nTest Set Results:")
print(f"Precision:  {metrics.box.mp:.4f}")
print(f"Recall:     {metrics.box.mr:.4f}")
print(f"mAP50:      {metrics.box.map50:.4f}")
print(f"mAP50-95:   {metrics.box.map:.4f}")
