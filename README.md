# LiDAR Object Classification

Object segmentation and classification in LiDAR point clouds using the DALES dataset, geometric feature extraction, PCA descriptors, DBSCAN clustering, and Random Forest classification.

The main part of the project is based on the DALES dataset. Additional UAV LiDAR files are used only as an experimental qualitative test of the trained model on unseen LAS data.

---

## Project Overview

This project consists of two parts:

1. **DALES-based object classification**
   - Load annotated DALES `.ply` point clouds.
   - Extract object-level geometric and PCA-based features.
   - Train several machine learning models.
   - Evaluate classification quality.
   - Analyze Random Forest feature importance.
   - Perform DBSCAN-based segmentation and visualization on DALES scenes.

2. **Transfer experiment on UAV LAS files**
   - Load independent `.las` point clouds.
   - Segment objects using DBSCAN.
   - Extract the same feature set as for DALES.
   - Classify clusters using the Random Forest model trained on DALES.
   - Visualize predicted classes qualitatively.

---

## Classes

The project uses the DALES semantic classes:

| ID | Class |
|---:|---|
| 1 | ground |
| 2 | vegetation |
| 3 | cars |
| 4 | trucks |
| 5 | power_lines |
| 6 | fences |
| 7 | poles |
| 8 | buildings |

---

## Feature Set

For every object or cluster, the following features are extracted:

- number of points
- length
- width
- height
- bounding box volume
- point density
- minimum height
- maximum height
- mean height
- height standard deviation
- intensity mean
- intensity standard deviation
- linearity
- planarity
- scattering
- omnivariance
- anisotropy
- eigenentropy
- verticality

---

## Requirements

- Python 3.10+
- NumPy
- Pandas
- Scikit-learn
- Open3D
- Matplotlib
- Plyfile
- LasPy
- Joblib
- tqdm

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

### DALES

The DALES dataset is not included in this repository.

Place it in:

```text
DALESObjects/
```

Expected example path:

```text
DALESObjects/train/5080_54435_new.ply
```

### UAV LAS data

The UAV LAS files are also not included because of their size.

Expected local structure:

```text
obloty/
├── cloud0.las
├── cloud1.las
└── cloud2.las
```

---

## DALES Pipeline

### 1. Check dataset structure

```bash
python src/01_check_dataset.py
```

### 2. Extract features from DALES training and test sets

```bash
python src/02_extract_features.py
python src/02_extract_features_test.py
```

Generated files:

```text
data/features_train.csv
data/features_test.csv
```

### 3. Train machine learning models

```bash
python src/03_train_models.py
```

The script compares:

- Random Forest
- Decision Tree
- KNN
- Logistic Regression
- SVM

The Random Forest model is saved to:

```text
models/random_forest.pkl
```

### 4. Analyze Random Forest

```bash
python src/04_random_forest_analysis.py
```

This generates feature importance and confusion matrix outputs in:

```text
results/
```

### 5. DBSCAN experiments on DALES

```bash
python src/05_dbscan_experiment.py
python src/06_dbscan_cluster_analysis.py
python src/07_visualize_dbscan.py
```

These scripts test DBSCAN segmentation and analyze cluster quality on a selected DALES scene.

### 6. Classify DALES DBSCAN clusters

```bash
python src/08_classify_dbscan_clusters.py
python src/09_visualize_predicted_classes.py
```

These scripts classify DBSCAN clusters using the trained Random Forest model and visualize the predicted classes.

### 7. Export DALES semantic top view

```bash
python src/10_export_dales_top_view.py
```

This creates a top-down visualization of a DALES scene colored by semantic class.

---

## UAV LAS Transfer Experiment

This part applies the model trained on DALES to independent UAV LiDAR scans.

No ground-truth labels are available for the UAV data, so results are evaluated visually.

### 1. Extract LAS cluster features

```bash
python src/11_extract_las_cluster_features.py cloud0.las
python src/11_extract_las_cluster_features.py cloud1.las
python src/11_extract_las_cluster_features.py cloud2.las
```

Generated files:

```text
data/las_cluster_features_cloud0.csv
data/las_cluster_features_cloud1.csv
data/las_cluster_features_cloud2.csv

data/las_cluster_points_labels_cloud0.npz
data/las_cluster_points_labels_cloud1.npz
data/las_cluster_points_labels_cloud2.npz
```

### 2. Inspect selected clusters

```bash
python src/12_visualize_selected_clusters.py cloud2.las 1
python src/12_visualize_selected_clusters.py cloud2.las 1 7
```

This is useful for manually checking individual DBSCAN clusters, for example potential power line or pole clusters.

### 3. Classify LAS clusters using the DALES model

```bash
python src/13_classify_las_clusters_rf.py cloud0
python src/13_classify_las_clusters_rf.py cloud1
python src/13_classify_las_clusters_rf.py cloud2
```

Generated files:

```text
data/las_cluster_predictions_cloud0.csv
data/las_cluster_predictions_cloud1.csv
data/las_cluster_predictions_cloud2.csv
```

### 4. Visualize LAS predictions

```bash
python src/14_visualize_las_predictions.py cloud0
python src/14_visualize_las_predictions.py cloud1
python src/14_visualize_las_predictions.py cloud2
```

This visualizes DBSCAN clusters colored by predicted DALES class.

### 5. Export RGB top views of LAS files

```bash
python src/15_export_las_top_view.py cloud0.las
python src/15_export_las_top_view.py cloud1.las
python src/15_export_las_top_view.py cloud2.las
```

Generated files:

```text
results/cloud0_top_view.png
results/cloud1_top_view.png
results/cloud2_top_view.png
```