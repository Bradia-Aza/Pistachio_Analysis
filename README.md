# Pistachio Dataset Analysis and Modeling

This project presents a comprehensive machine learning pipeline built around the **Pistachio Image and Feature Dataset**. It includes preprocessing, feature selection, dimensionality reduction, classification, outlier detection, and clustering using both handcrafted and CNN-extracted features.

## 📁 Dataset

The dataset is from [Murat Koklu's website](https://www.muratkoklu.com/datasets/), containing:
- 28 handcrafted features (morphological, shape, color)
- Pistachio images (600x600 RGB)

---

## 📌 Project Overview

### 1. **Data Loading & Exploration**
- Loaded `.xlsx` and image data using Pandas and TensorFlow
- Explored statistical summaries, class distributions, and correlation heatmaps

### 2. **Feature Engineering**
- Correlation filtering
- Recursive Feature Elimination (RFE) with Logistic Regression and SVM
- Dimensionality reduction using **PCA** and **LDA**

### 3. **Preprocessing**
- Implemented a `DataPreprocessor` class for:
  - Standardization / Normalization
  - SMOTE for imbalance
  - Encoding
  - Train/test splitting

### 4. **Feature Extraction from Images**
- Used **VGG16** pretrained model to extract deep features
- Built and trained **3 custom CNNs**
- Saved features using NumPy

### 5. **Classification**
- Implemented a `ClassifierEvaluator` class to evaluate:
  - Naïve Bayes
  - Support Vector Machine (SVM)
  - Multi-Layer Perceptron (MLP)
  - Stacking Ensemble
- Hyperparameter tuning using **GridSearchCV**

### 6. **Outlier Detection**
- Created an `OutlierDetector` class using:
  - Local Outlier Factor (LOF)
  - Isolation Forest (ISF)
  - One-Class SVM (OCSVM)
- Visualized results using 1D/2D/3D plots (LDA & PCA)

### 7. **Clustering**
- Built a `ClusteringEvaluator` class supporting:
  - **KMeans**
  - **DBSCAN**
  - **EM (Gaussian Mixture)**
- Evaluated with:
  - Silhouette Score
  - Calinski-Harabasz Index
  - Davies-Bouldin Index

---

## 🧠 Models Trained On
- Raw features
- Cleaned feature subsets
- PCA / LDA transformed data
- Deep features from:
  - Custom CNNs
  - VGG16

---

## 📊 Visualizations
- Correlation heatmaps
- PCA/LDA projections
- Outlier and cluster plots in 2D & 3D

---

## 📂 Structure
```bash
Pistachio_Dataset/
├── preprocessing/
├── classification/
├── cnn_models/
├── outlier_detection/
├── clustering/
├── extracted_features/
└── pistachio_df_cleaned.xlsx
```
