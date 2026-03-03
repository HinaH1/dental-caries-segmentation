# Experiment Log

---

## Experiment 1: Object Detection Approach

### Objective
Initial attempt using YOLO object detection for identifying dental caries.

### Outcome
- Model trained using bounding box detection.
- Dataset size identified as insufficient for robust performance.
- Segmentation deemed more suitable due to irregular lesion boundaries.

### Conclusion
Object detection approach was abandoned in favour of instance segmentation.

---

## Experiment 2: Clinical Dataset Review

### Objective
Evaluate dataset suitability for real-world clinical relevance.

### Findings
- Initial dataset consisted primarily of periapical radiographs.
- Consultation with dental professional indicated that bitewing radiographs are more commonly used clinically for caries detection.

### Conclusion
Dataset limitations identified.
Further dataset refinement and validation performed.

---

## Experiment 3: Segmentation-Based Approach (Current)

### Objective
Implement YOLOv8 instance segmentation for improved lesion boundary modelling.

### Status
Ongoing.
Dataset validated and annotation sanity checks completed.
Baseline training to follow.
