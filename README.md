# Precision Agriculture Spray Quality Assessment

## Project Overview

This project aims to develop a smartphone application for precision spray quality assessment in agriculture. By leveraging computer vision and machine learning techniques, we address the limitations of current portable spray quality assessors, accurately detecting overlapping droplets on water-sensitive paper (WSP).

## Key Features

- Dual-method framework combining classical computer vision and machine learning for WSP and droplet segmentation
  1. Segmentation and perspective correction of WSP
  2. Detection and segmentation of individual droplets in WSP
- Computation of spray quality metrics (Volume Median Diameter (VMD), Relative Span Factor (RSF) and Coverage Percentage)
- Algorithm for automatic creation and annotation of a synthetic dataset
- Android application with a web service that enables real-time image processing and statistical analysis;

<div align="center">
	<img src="https://github.com/user-attachments/assets/090141aa-e757-4783-8bd7-b084385ebd04" width="50%" >
</div>

## Repository Structure

```
spray-quality-assessment/
├── data/
│   ├── droplet/
│       ├── synthetic/
│       └── real/
│   └── paper/
├── models/
│   ├── droplet/
│       ├── yolov8/
│       └── mrcnn/
│   └── paper/
├── results/
│   ├── evaluation/
│       ├── droplet/
│           ├── synthetic/
│           ├── real/
│           └── general/
│       └── paper/
│   └── latex/
├── src/
│   ├── common/
│   ├── Segmentation/
│       ├── dataset/
│       ├── droplet/
│           ├── ccv/
│           └── cnn/
│               ├── Cellpose/
│               ├── Mask R-CNN/
│               └── YOLOv8/
│       ├── evaluate_algorithms/
│       └── paper/
│           ├── ccv/
│           └── cnn/
│   ├── SyntheticDataset/
│   └── WebService/
├── docs/
├── app/
└── README.md
```

- `data/`: Synthetic and real datasets for WSP and droplet segmentation
- `models/`: Trained models for each task
- `results/`:
  - `evaluation/`: Performance evaluation metrics CSV files
  - `latex/`: Files used for the development of the final Dissertation document
- `src/`: Source code for the core functionality
  - `Common/`: Scripts and classes common to multiple folders, including a file path
  - `Segmentation/`: Droplet segmentation algorithms
    - `dataset/`: Script for pre-annotation of the real WSP images
    - `droplet/`: Scripts for droplet segmentation
    - `evaluate_algorithms/`: Scripts to evaluate each one of the methods developed with different datasets
    - `paper/`: Scripts for WSP segmentation
  - `SyntheticDataset/`: Algorithm for creation of synthetic dataset
  - `WebService/`: Script to launch the server of the web service implemented to apply the segmentation algorithms to image received 
- `app/`: Smartphone application code
- `README.md`: This file, providing an overview of the project

There are three different `requirements.txt` files, given that each model requires different library versions. They can be found inside the Segmentation folders along with the models' scripts.

## Datasets

### Synthetic Dataset
- Programmatically generated with configurable parameters
- Mimics real conditions (resolution, background color gradient, droplet size distribution)
- Currently 300 images across three different resolutions


<div align="center">
  <p float="center">
    <img src="https://github.com/user-attachments/assets/e0ed35ac-4cdb-47a4-ad9f-8f7a3b3f09e4" width="100" />
    <img src="https://github.com/user-attachments/assets/eb6efb17-7ad7-4d08-ab00-dfbe20598d86" width="100" /> 
    <img src="https://github.com/user-attachments/assets/fc8527e7-70e8-4d86-811b-906f159c8f22" width="100" />
    <img src="https://github.com/user-attachments/assets/16db7dad-35de-4b05-8019-6d740d0cd8a5" width="100" />
    <img src="https://github.com/user-attachments/assets/71c12005-81c3-452f-b202-b749c9c2f7eb" width="100" />
    <img src="https://github.com/user-attachments/assets/b0d8bf62-529a-46be-a2d7-eef2a6cade76" width="100" />
  </p>
</div>





### Real Dataset for WSP
- Images of real WSP from two different sources
- Images of printed synthetic papers in multiple real-world backdrops of green foliage

### Real Dataset for Droplets
- Two images from datasets provided

## Methods and Results

### Water Sensitive Paper Segmentation
- YOLOv8 achieved an average Intersection over Union of 97.76%

### Droplet Segmentation
1. Machine Learning Models:
   - CellPose: Highest precision (96.18%) in droplet detection, especially for overlapping droplets
   - YOLOv8 and Mask R-CNN: Robust performance, slightly conservative in segmentation
2. Classical Computer Vision:
   - Provided a reliable baseline
   - Struggled with complex cases of overlapping droplets

<div align="center" style="display: flex; justify-content: center; flex-wrap: wrap;">
  <figure style="margin: 10px; text-align: center; width: 100px;">
    <figcaption style="margin-bottom: 5px;">YOLOv8</figcaption>
    <img src="https://github.com/user-attachments/assets/a04b376a-d403-47f2-a5be-869ac0bb44f9" width="100" />
  </figure>
  <figure style="margin: 10px; text-align: center; width: 100px;">
    <figcaption style="margin-bottom: 5px;">Cellpose</figcaption>
    <img src="https://github.com/user-attachments/assets/0c74a7d7-e6de-47b3-9a33-d87b6059bd48" width="100" /> 
  </figure>
  <figure style="margin: 10px; text-align: center; width: 100px;">
    <figcaption style="margin-bottom: 5px;">Mask R-CNN</figcaption>
    <img src="https://github.com/user-attachments/assets/41dcfd65-2d0c-44eb-b7de-68b99ebb1da4" width="100" />
  </figure>
  <figure style="margin: 10px; text-align: top; width: 100px;">
    <figcaption style="margin-bottom: 5px;">Classical CV</figcaption>
    <img src="https://github.com/user-attachments/assets/3f474061-e8ed-41dd-9b67-a398e7694d99" width="100" />
  </figure>
</div>

## Future Work

1. Data Enhancement:
   - Implement advanced data augmentation techniques
   - Expand the real-world dataset
   - Explore automated or semi-automated annotation methods

2. Model Improvement:
   - Optimize machine learning models (YOLOv8, Mask R-CNN)
   - Retrain Cellpose with the synthetic dataset
   - Develop hybrid approaches combining classical and ML techniques

3. Overlapping Droplet Handling:
   - Investigate advanced techniques for improved segmentation of overlapping droplets

4. Real-world Validation:
   - Conduct extensive field tests across various crops and environmental conditions



