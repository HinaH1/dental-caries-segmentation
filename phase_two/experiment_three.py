"""
experiment_three.py

This script runs Experiment Three of Phase Two - YOLO26-seg instance 
segmentation on the CariXray dataset.

This experiment builds on from experiment two this time instead running YOLO 
version 26 on the dataset


A different seed to Experiment two is used as each experiment is treated 
as an independent training run. The results are evaluated on the same 
held-out test set to allow direct comparison with Experiment Two and 
against the Dang et al. benchmark.

Author: HinaH1
"""


import random
import numpy as np
import torch
from ultralytics import YOLO

# Reproducibility
SEED = 54

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")

# Load YOLO26 segmentation model
model = YOLO("yolo26n-seg.pt")

print("YOLO26 segmentation model loaded successfully")

# Train
results = model.train(
    data="/usr/tmp/final_year_project/SEGMENTATION/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    seed=SEED,
    device=0,
    project="/var/tmp/final_year_project/EXPERIMENT3",
    name="yolo26n_caries_seg",
    exist_ok=True,
)

print("Training complete")

# Load the best weights saved during training and evaluate on the test set
# The test set was kept completely separate from training and validation
model = YOLO('/usr/tmp/final_year_project/EXPERIMENT3/yolo26n_caries_seg/weights/best.pt')

metrics = model.val(
    data='/usr/tmp/final_year_project/SEGMENTATION/SEGMENTATION/data.yaml',
    split='test'
)

print("\nTest Set Results:")
print(f"Precision(B):  {metrics.box.mp:.4f}")
print(f"Recall(B):     {metrics.box.mr:.4f}")
print(f"mAP50(B):      {metrics.box.map50:.4f}")
print(f"mAP50-95(B):   {metrics.box.map:.4f}")
print(f"Precision(M):  {metrics.seg.mp:.4f}")
print(f"Recall(M):     {metrics.seg.mr:.4f}")
print(f"mAP50(M):      {metrics.seg.map50:.4f}")
print(f"mAP50-95(M):   {metrics.seg.map:.4f}")
