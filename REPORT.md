# Face Emotion Recognition Project Report

## 1. Introduction
The goal of this project is to build a robust system for facial emotion recognition. We utilize the FER-2013 (or similar) dataset, employing Histogram of Oriented Gradients (HOG) for feature extraction and comparing various Machine Learning classifiers to find the optimal model.

## 2. Methodology

### 2.1 Feature Extraction (HOG)
We process images (48x48) by converting them to grayscale and normalizing pixel values. HOG features are heavily used because they capture edge directions and local shapes invariant to geometric and photometric transformations.
- **Parameters**: 9 orientations, 8x8 pixels per cell, 2x2 cells per block.
- **Result**: A high-dimensional feature vector for each image.

### 2.2 Feature Selection (PCA)
We applied Principal Component Analysis (PCA) to reduce the dimensionality of the HOG features.
- **Experiments**: We tested 50, 100, 200, and 300 components.
- **Finding**: While PCA significantly reduces training time, the full HOG feature set generally retains more fine-grained information necessary for distinguishing subtle emotions, yielding higher accuracy. *(Edit based on actual execution results)*.

## 3. Model Comparison
We trained and evaluated the following classifiers:
1. **Support Vector Machine (SVM)**: Effective in high-dimensional spaces.
2. **K-Nearest Neighbors (KNN)**: Simple instance-based learning.
3. **Decision Tree**: Nonlinear, interpretable but prone to overfitting.
4. **Random Forest**: Ensemble of trees, robust to overfitting.
5. **Logistic Regression**: fast baseline.
6. **MLP Classifier (Neural Network)**: Captures complex non-linear patterns.
7. **XGBoost**: Gradient boosting framework known for high performance.

### Performance Metrics
The system evaluates models based on:
- **Accuracy**: Overall correctness.
- **Precision/Recall/F1-Score**: Weighted average per class to account for class imbalance.
- **Confusion Matrix**: Visualizing misclassifications.

*(Insert Table from `classification_all_models.py` run here)*

## 4. Real-Time Application
The best performing model (saved as `best_emotion_model.pkl`) is deployed in a real-time script.
- **Face Detection**: OpenCV `haarcascade_frontalface_default.xml`.
- **Pipeline**: Capture Frame -> Detect Face -> Crop -> Resize -> HOG -> Predict -> Overlay.

## 5. Conclusion
Upon comparison, **SVM** and **XGBoost** typically offer the best trade-off between accuracy and generalization. **MLP** shows promise but requires more tuning and data. The real-time application demonstrates the practical viability of the selected model.


