# Dental Caries Segmentation Using YOLOv8

## Project Overview

This project investigates the use of deep learning for automatic segmentation of dental caries in periapical radiographs using Ultralytics YOLOv8.

The aim is to develop and evaluate a computer vision model capable of identifying carious lesions from radiographic images, supporting potential applications in computer-aided dental diagnostics.

---

## Research Objectives

- Validate and analyse dataset structure  
- Perform annotation sanity checks  
- Investigate dataset label distribution and class imbalance  
- Train a baseline YOLOv8 segmentation model  
- Evaluate model performance using mAP, precision, recall and IoU  
- Experiment with hyperparameter optimisation and model improvements  

---

## Dataset

The dataset consists of periapical radiographs annotated for dental caries using polygon segmentation masks.

Preprocessing steps:
- Removal of non-relevant classes (restoration class removed)  
- Verification of YOLOv8 segmentation format  
- Label distribution analysis  
- Annotation alignment validation  

Class distribution analysis identified:
- Total images: 757  
- Images containing caries: 343  
- Images without caries: 414  

This class imbalance is considered during model evaluation.

---

## Model Architecture

- Framework: Ultralytics YOLOv8  
- Task: Instance Segmentation  
- Input: Periapical radiographs  
- Output: Polygon segmentation masks identifying carious lesions  

YOLOv8 was selected due to:
- Strong performance in real-time detection and segmentation tasks  
- Efficient training pipeline  
- Built-in evaluation metrics (mAP50, mAP50-95, Precision, Recall)  

---

## Project Structure

```
dental-caries-segmentation/
│
├── notebooks/
│   └── dental_caries_segmentation.ipynb
│
├── experiments/
│   └── experiment_log.md
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Methodology

1. Dataset acquisition and validation  
2. Dataset structure verification  
3. Label distribution analysis  
4. Annotation sanity check (polygon-to-pixel validation)  
5. Baseline YOLOv8 segmentation training  
6. Performance evaluation  
7. Iterative experimentation  

---

## Reproducibility

To install required dependencies:

```bash
pip install -r requirements.txt
```

To train using YOLOv8 (example):

```bash
yolo segment train data=data.yaml model=yolov8n-seg.pt epochs=100 imgsz=640
```

---

## Author

Hina Habib  
BSc Computer Science  
Final Year Dissertation  
2026  

---

## License

This project is licensed under the MIT License.
