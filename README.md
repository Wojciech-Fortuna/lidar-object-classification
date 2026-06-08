# LiDAR Object Classification

A project for object segmentation and classification in LiDAR point clouds using geometric features, DBSCAN clustering, PCA-based descriptors, and Random Forest classifiers.

## Project Overview

This project uses the DALES (Dayton Annotated LiDAR Earth Scan) dataset to:

* Extract geometric features from segmented objects.
* Train machine learning models for object classification.
* Analyze feature importance.
* Segment point clouds using DBSCAN.
* Classify segmented clusters using a trained Random Forest model.
* Visualize segmentation and classification results.

## Pipeline

    DALES Dataset
          │
          ▼
    Feature Extraction
          │
          ▼
    Model Training
          │
          ▼
    Random Forest Classifier
          │
          ▼
    DBSCAN Segmentation
          │
          ▼
    Cluster Feature Extraction
          │
          ▼
    Object Classification
          │
          ▼
    Visualization

## Classes

The project works with the following object categories:

* Ground
* Vegetation
* Cars
* Trucks
* Power Lines
* Fences
* Poles
* Buildings

## Requirements

* Python 3.10+
* NumPy
* Pandas
* Scikit-learn
* Open3D
* Matplotlib
* Plyfile

Install dependencies:

    pip install -r requirements.txt

## Dataset

The DALES dataset is not included in this repository.  
Download the dataset separately and place it in:

    DALESObjects/

## Usage

### 1. Extract Features
    python src/02_extract_features.py
    python src/02_extract_features_test.py

### 2. Train Models
    python src/03_train_models.py

### 3. Analyze Random Forest
    python src/04_random_forest_analysis.py

### 4. DBSCAN Segmentation
    python src/05_dbscan_experiment.py

### 5. Cluster Classification
    python src/08_classify_dbscan_clusters.py

### 6. Visualization
    python src/07_visualize_dbscan.py
    python src/09_visualize_predicted_classes.py

## Notes

* Large datasets and generated results are excluded from version control using `.gitignore`.