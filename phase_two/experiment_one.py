"""
experiment_one.py

This script runs Experiment One of Phase Two — YOLOv8 object detection 
on the CariXray dataset.

The purpose of this experiment is to establish a detection baseline on 
CariXray using the same dataset and metrics as Dang et al., so that 
the results can be compared directly against their published benchmark.
A fixed seed is used throughout to make sure the results are reproducible.

After training, the best model weights are evaluated on the held-out 
test set and the key metrics are printed.

Author: HinaH1
"""

import random
import numpy as np
import torch
from ultralytics import YOLO

# Fix random seeds to make the experiment reproducible
# If this script is run again with the same seed the results should be identical
random.seed(23)
np.random.seed(23)
torch.manual_seed(23)

# Print environment info to confirm the GPU is available before training starts
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

## Load the YOLOv8 nano detection model
# This was chosen as it is the most efficient configuration and suited to the scale of the dataset
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='/usr/tmp/trial/DETECTION/DETECTION/data.yaml',
    epochs=200,
    imgsz=640,
    batch=8,
    patience = 20,
    seed=23,
    device=0,
    project='/usr/tmp/trial/EXPERIMENT1',
    name='yolov8n_caries_detection',
    exist_ok=True
)

print("Training complete")

# Load the best weights saved during training and evaluate on the test set
# The test set was kept completely separate from training and validation
model = YOLO('/usr/tmp/final_year_project/EXPERIMENT1/yolov8n_caries_detection/weights/best.pt')

metrics = model.val(
    data='/usr/tmp/trial/DETECTION/DETECTION/data.yaml',
    split='test'
)

# Print the test set results
print("\nTest Set Results:")
print(f"Precision:  {metrics.box.mp:.4f}")
print(f"Recall:     {metrics.box.mr:.4f}")
print(f"mAP50:      {metrics.box.map50:.4f}")
print(f"mAP50-95:   {metrics.box.map:.4f}")
