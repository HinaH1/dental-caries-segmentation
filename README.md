# Dental Caries Detection and Segmentation Using YOLO

## Project Overview

This project investigates the use of deep learning for automatic detection and segmentation of dental caries in periapical radiographs. Three models are developed and compared: a YOLOv8 object detection model, a YOLOv8 instance segmentation model, and a YOLO26 instance segmentation model.

The project is structured in two phases. Phase One develops and validates the experimental pipeline on open-source datasets. Phase Two applies that pipeline to CariXray, a clinically validated periapical X-ray dataset, enabling direct comparison against a published benchmark.

---

## Research Objectives

- Develop a YOLOv8 object detection model to establish a bounding box detection baseline for caries localisation
- Develop a YOLOv8-seg instance segmentation model to produce pixel-level lesion masks
- Develop a YOLO26-seg instance segmentation model and compare its performance against YOLOv8-seg
- Evaluate and compare all three models against the published CariXray benchmark of Dang et al.
- Identify limitations and propose directions for future work

---

## Project Structure

```
dental-caries-segmentation/
│
├── phase_one/
│   ├── initial_experiment_one_ipnyb.ipynb
│   ├── experiment_two_YOLOv8_seg.ipynb
│   └── experiment_three_YOLO26_seg.ipynb
│
├── phase_two/
│   ├── experiment_one.py
│   ├── experiment_two.py
│   ├── experiment_three.py
│   └── data_setup/ 
│          ├── detection_data_setup.py
│          └──segmentation_data_setup.py
│ 
│           
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Datasets

### Phase One: Open-Source Datasets

Phase One used two open-source datasets sourced from Roboflow and Kaggle to develop the pipeline before the clinically validated dataset was available.

**Experiment One — Kaggle Dental Cavity Radiograph Images**
- Source: [Kaggle — alokkumar175358](https://www.kaggle.com/datasets/alokkumar175358/dental-cavity-radiograph-images)
- Task: Object detection (bounding boxes)
- Purpose: Establish a working detection pipeline and confirm the limitations of bounding box outputs for caries localisation

**Experiments Two and Three — tumverilerdeneme (Roboflow)**
- Source: [Roboflow — University workspace](https://universe.roboflow.com/university-ciekc/tumverilerdeneme)
- Task: Instance segmentation (polygon masks)
- Notes: Clinically reviewed and approved before use. The Restoration class was removed to isolate caries annotations only. This dataset is one of the sources used by Dang et al. in constructing CariXray.

### Phase Two: CariXray

- Source: Dang et al., *Machine Vision and Applications*, 2025. DOI: [10.1007/s00138-025-01776-8](https://doi.org/10.1007/s00138-025-01776-8)
- ~4,700 periapical radiographs annotated for dental caries under dental expert supervision
- Supports both bounding box detection and polygon segmentation tasks
- Split: 3,299 training / 500 validation / 358 test images
- Selected to enable direct benchmarking against published results

> CariXray access was obtained directly from the authors. To reproduce Phase Two experiments, contact Dang et al. to request dataset access.

---

## Methodology

The project followed a two-phase iterative structure rather than a fixed sprint-based approach, allowing each experiment to inform the next.

### Phase One: Pipeline Development

Three experiments were conducted on open-source datasets to develop the training pipeline and validate the approach before progressing to the main phase.

1. **Experiment One — YOLOv8 Detection:** Trained on the Kaggle dataset for 50 epochs at 640px. Confirmed that bounding box outputs are insufficient for representing the irregular boundaries of carious lesions, motivating the move to segmentation.

2. **Experiment Two — YOLOv8n-seg:** Trained on tumverilerdeneme. A baseline run (50 epochs, 640px) was followed by an improved run (100 epochs max, 768px, early stopping patience 20) after the baseline showed the model had not yet converged at epoch 50.

3. **Experiment Three — YOLO26n-seg:** Trained on tumverilerdeneme using the same two-stage approach. The higher starting classification loss at epoch 1 (11.04 vs 6.39 for YOLOv8-seg) confirmed YOLO26 requires more epochs to settle, reinforcing the extended training configuration used in Phase Two.

### Phase Two: Main Experiments on CariXray

Three experiments were conducted on CariXray using the configuration informed by Phase One findings.

1. **Experiment One — YOLOv8n Detection**
2. **Experiment Two — YOLOv8n-seg**
3. **Experiment Three — YOLO26n-seg**

All Phase Two experiments used 100 epochs, 640px image size, batch size 16, and no early stopping, matching the Dang et al. benchmark configuration to ensure comparability.

---

## Model Architectures

| Model | Task | Variant | Rationale |
|---|---|---|---|
| YOLOv8n | Object Detection | Nano | Establishes detection baseline; matches Dang et al. config |
| YOLOv8n-seg | Instance Segmentation | Nano | Strong literature performance; evaluated in Dang et al. benchmark |
| YOLO26n-seg | Instance Segmentation | Nano | Most recent YOLO architecture; not previously evaluated in dental imaging |

All models were accessed through the [Ultralytics](https://github.com/ultralytics/ultralytics) framework (v8.4.37).

---

## Training Configuration

### Phase One

| Parameter | Baseline (Exp 2 & 3) | Improved (Exp 2 & 3) |
|---|---|---|
| Epochs | 50 | 100 (max) |
| Image size | 640px | 768px |
| Batch size | 4 | 4 |
| Early stopping | No | Yes, patience 20 |
| Environment | Google Colab (T4 GPU) | Google Colab (T4 GPU) |

### Phase Two

| Parameter | All Experiments |
|---|---|
| Epochs | 100 |
| Image size | 640px |
| Batch size | 16 |
| Early stopping | No |
| Environment | NVIDIA RTX 4070 (12GB VRAM), Python 3.9, CUDA 13.1 |

Random seeds: Experiment One = 42, Experiment Two = 123, Experiment Three = 54.

---

## Evaluation Metrics

All experiments were evaluated using the same five metrics reported by Dang et al.:

- **Precision** — fraction of positive predictions that are correct
- **Recall** — fraction of true lesions successfully detected
- **F1-score** — harmonic mean of Precision and Recall
- **mAP@0.5** — mean Average Precision at 50% IoU overlap threshold
- **mAP@0.5-0.95** — mean Average Precision averaged across IoU thresholds from 50% to 95%

All models were evaluated on the held-out test set (358 images), which played no part in any training or configuration decisions.

---

## Reproducibility

### Phase One

Install dependencies and run notebooks in order:

```bash
pip install -r requirements.txt
```

Phase One notebooks are self-contained and run on Google Colab. Open each notebook and run cells in sequence. The Roboflow API key must be added to Colab Secrets before running.

### Phase Two

Phase Two requires the CariXray dataset, which must be requested directly from Dang et al. Once obtained, run the preparation scripts to reorganise the data into YOLO format, please not the location of the files must be updated manually within the script prior to running:

```bash
python data_preparation/prepare_detection.py
python data_preparation/prepare_segmentation.py
```

Then train using Ultralytics (example for Experiment Two):

```bash
yolo segment train data=data_fixed.yaml model=yolov8n-seg.pt epochs=100 imgsz=640 batch=16 seed=123
```

All scripts, configuration files, training logs, and results CSVs needed to reproduce all six experiments are available in this repository.

---

## Author

Hina Habib  
BSc Computer Science (Digital & Technology Solutions)  
COMP3932 Synoptic Project  
University of Leeds, 2025/2026

---

## License

This project is licensed under the MIT License.
