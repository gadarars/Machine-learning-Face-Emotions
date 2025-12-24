# Facial Emotion Recognition System

A comprehensive Machine Learning project implementing real-time facial emotion detection using classical ML techniques with HOG features and multiple classifiers.

---

## 📋 Project Overview

This project classifies seven facial emotions: **Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise** using:
- **Feature Extraction**: Histogram of Oriented Gradients (HOG)
- **Feature Selection**: Principal Component Analysis (PCA)
- **Classifiers**: SVM, KNN, Random Forest, Decision Tree, Logistic Regression, MLP Neural Network, XGBoost
- **Real-time Detection**: Webcam-based emotion recognition

**Dataset**: FER2013 (Facial Expression Recognition)
- Training images: 28,709
- Testing images: 7,178
- Image size: 48×48 grayscale

---

## 📂 Project Structure
```
Machine-learning-Face-Emotions/
├── train/                              # Training dataset (7 emotion folders)
├── test/                               # Testing dataset (7 emotion folders)
├── models/                             # Saved trained models
│   ├── best_emotion_model.pkl          # Best performing model (XGBoost)
│   └── best_pca.pkl                    # Optimal PCA transformer
├── results/                            # Visualization outputs
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   ├── pca_comparison.png
│   └── metrics_comparison.png
├── load_visualize.py                   # Dataset exploration & visualization
├── preprocessing.py                    # Data loading & preprocessing
├── pca_feature_selection.py           # PCA feature selection analysis
├── classification_all_models.py       # Train & evaluate all models
├── realtime_emotion_detection.py      # Real-time webcam emotion detection
├── visualize_results.py               # Generate result visualizations
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

---

## 🚀 Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required libraries**:
- numpy
- opencv-python
- scikit-learn
- scikit-image
- matplotlib
- seaborn
- xgboost
- joblib

### 2. Dataset Setup

Place your training and testing datasets in the following structure:
```
train/
  ├── angry/
  ├── disgust/
  ├── fear/
  ├── happy/
  ├── neutral/
  ├── sad/
  └── surprise/

test/
  └── (same structure as train)
```

---

## 📊 How to Run

### Step 1: Explore Dataset (Optional)
```bash
python load_visualize.py
```
Displays sample images from each emotion class.

---

### Step 2: Data Preprocessing
```bash
python preprocessing.py
```
**What it does**:
- Loads images from train/test folders
- Converts to grayscale (48×48)
- Normalizes pixel values (0-1)
- Creates label mappings

**Output**: Prints data shapes and confirms preprocessing completion.

---

### Step 3: Feature Selection with PCA
```bash
python pca_feature_selection.py
```
**What it does**:
- Extracts HOG features from all images
- Tests PCA with different components (50, 100, 200, 300)
- Compares accuracy and explained variance
- Saves best PCA model

**Key Results**:
- **Baseline (Full HOG - 900 features)**: 44.47% accuracy
- **PCA 50 components**: 54.25% accuracy
- **PCA 100 components**: 54.86% accuracy ⭐ **BEST**
- **PCA 200 components**: 54.17% accuracy
- **PCA 300 components**: 52.93% accuracy

**Saved**: `models/best_pca.pkl`

---

### Step 4: Train & Compare All Models
```bash
python classification_all_models.py
```
**What it does**:
- Loads/caches HOG features (first run takes ~15 min, subsequent runs ~2 min)
- Trains 7 classifiers
- Evaluates performance metrics
- Saves best model

**Output**: Creates `models/best_emotion_model.pkl`

---

### Step 5: Generate Visualizations
```bash
python visualize_results.py
```
**What it does**:
- Creates confusion matrix heatmap
- Generates model comparison bar chart
- Plots PCA analysis results
- Compares Precision, Recall, F1-Score

**Output**: Saves 4 charts to `results/` folder

---

### Step 6: Real-Time Emotion Detection
```bash
python realtime_emotion_detection.py
```
**What it does**:
- Opens webcam feed
- Detects faces using Haar Cascade
- Extracts HOG features in real-time
- Predicts emotion with confidence percentage
- Displays colored bounding box (green for positive, red for negative emotions)

**Controls**: Press **'q'** to quit

---

## 📈 Results Summary

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Training Time (s) |
|-------|----------|-----------|--------|----------|-------------------|
| **XGBoost** ⭐ | **49.94%** | **49.77%** | **49.94%** | **49.04%** | 209.31 |
| KNN | 46.71% | 46.50% | 46.71% | 45.05% | 0.03 |
| Random Forest | 46.32% | 47.74% | 46.32% | 44.02% | 109.40 |
| MLP (Neural Net) | 45.65% | 45.73% | 45.65% | 45.65% | 280.16 |
| SVM (Linear) | 44.47% | 43.13% | 44.47% | 43.33% | 8946.37 |
| Logistic Regression | 44.09% | 42.16% | 44.09% | 42.56% | 36.94 |
| Decision Tree | 32.66% | 32.79% | 32.66% | 32.70% | 50.45 |

**Best Model**: XGBoost with **49.94% accuracy**

---

### Feature Selection Impact

| Method | Features | Explained Variance | Accuracy | Improvement |
|--------|----------|-------------------|----------|-------------|
| Full HOG | 900 | 100% | 44.47% | Baseline |
| PCA 50 | 50 | 54.83% | 54.25% | +9.78% |
| **PCA 100** ⭐ | **100** | **70.64%** | **54.86%** | **+10.39%** |
| PCA 200 | 200 | 88.22% | 54.17% | +9.70% |
| PCA 300 | 300 | 96.93% | 52.93% | +8.46% |

**Key Finding**: PCA with 100 components provides the best balance, capturing 70.64% variance while improving accuracy by over 10%.

---

## 🎯 Key Features Implemented

✅ **Dataset**: Split into training (28,709) and testing (7,178) sets  
✅ **Feature Extraction**: HOG (Histogram of Oriented Gradients)  
✅ **Feature Selection**: PCA with multiple component analysis  
✅ **Multiple Classifiers**: 7 different algorithms tested  
✅ **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix  
✅ **Model Persistence**: Best model saved for reuse  
✅ **Real-time Detection**: Webcam-based emotion recognition  
✅ **Visualization**: Professional charts and confusion matrices  

---

## 💡 Technical Details

### Feature Extraction (HOG)
- **Orientations**: 9
- **Pixels per cell**: 8×8
- **Cells per block**: 2×2
- **Block normalization**: L2-Hys
- **Feature dimension**: 900 (before PCA)

### Best Model Configuration (XGBoost)
- **Algorithm**: Gradient Boosting
- **Accuracy**: 49.94%
- **Training time**: 209 seconds
- **Saved as**: `models/best_emotion_model.pkl`

### Real-time Detection Optimization
- **Face detection**: Haar Cascade (frontal face)
- **Processing**: Every 5th frame (5x speed boost)
- **Frame size**: Resized to 640×480 for performance
- **Latency**: ~100-200ms per prediction

---

## 🔬 Observations & Insights

### Strengths
1. **PCA significantly improved accuracy** from 44% to 55% with dimensionality reduction
2. **XGBoost outperformed** traditional classifiers
3. **Real-time detection** achieves smooth performance with frame skipping
4. **Comprehensive evaluation** with multiple metrics

### Challenges
1. **Class imbalance**: Model tends to predict "happy" and "neutral" more frequently (dataset bias)
2. **Similar emotions**: Angry/disgust and sad/neutral share similar facial features
3. **Accuracy ceiling**: 50% is typical for classical ML on emotion recognition
4. **Computational cost**: SVM training took 2.5 hours vs XGBoost's 3.5 minutes

### Future Improvements
- Use deep learning (CNN) for potentially higher accuracy (70-80%)
- Balance dataset with augmentation techniques
- Implement ensemble methods combining multiple classifiers
- Add temporal information for video-based emotion recognition

---

## 📝 Project Requirements Met

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| I. Dataset split | ✅ Complete | Train: 28,709 / Test: 7,178 |
| II. Feature extraction | ✅ Complete | HOG features (900 dimensions) |
| III. Feature selection | ✅ Complete | PCA (tested 50, 100, 200, 300) |
| IV. Multiple classifiers | ✅ Complete | 7 classifiers implemented |
| V. Performance evaluation | ✅ Complete | Accuracy, Confusion Matrix, Precision, Recall, F1 |
| Real-time detection | ✅ Bonus | Webcam-based emotion recognition |

---
## 📞 Troubleshooting

### Issue: "Model not found" error
**Solution**: Run `classification_all_models.py` first to train and save the model.

### Issue: Webcam not opening
**Solution**: 
- Check if another application is using the webcam
- Try changing camera index: `cv2.VideoCapture(1)` instead of `0`

### Issue: Slow first run
**Solution**: First run takes 10-15 minutes to extract HOG features. Subsequent runs use cached `.npy` files and complete in ~2 minutes.

### Issue: Poor emotion detection accuracy
**Solution**: 
- Ensure good lighting conditions
- Make exaggerated facial expressions
- 50% accuracy is normal for this approach
- Face directly toward camera

---

## 📚 References

- **Dataset**: FER2013 (Facial Expression Recognition Challenge)
- **HOG Features**: Dalal & Triggs (2005)
- **Libraries**: scikit-learn, OpenCV, XGBoost
- **Face Detection**: Viola-Jones Haar Cascade

---


**Last Updated**: December 24, 2025