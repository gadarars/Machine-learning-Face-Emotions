import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import seaborn as sns

from skimage.feature import hog
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             precision_score, recall_score, f1_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import time

# ==========================================
# CONFIGURATION
# ==========================================
TRAIN_DIR = "train"
TEST_DIR = "test"
IMG_SIZE = 48
MODELS_DIR = "models"
RESULTS_DIR = "results"
CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")
ROC_DIR = os.path.join(RESULTS_DIR, "roc_curves")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)
os.makedirs(ROC_DIR, exist_ok=True)

# Cache files
TRAIN_HOG_FILE = "X_train_hog.npy"
TEST_HOG_FILE = "X_test_hog.npy"
TRAIN_LABELS_FILE = "y_train.npy"
TEST_LABELS_FILE = "y_test.npy"

# ==========================================
# DATA LOADING
# ==========================================
def load_data_and_features():
    # Attempt to load cache
    if all(os.path.exists(f) for f in [TRAIN_HOG_FILE, TEST_HOG_FILE, TRAIN_LABELS_FILE, TEST_LABELS_FILE]):
        print("[INFO] Loading cached features...")
        X_train = np.load(TRAIN_HOG_FILE)
        X_test = np.load(TEST_HOG_FILE)
        y_train = np.load(TRAIN_LABELS_FILE)
        y_test = np.load(TEST_LABELS_FILE)
        
        # We need label map for display. Re-derive from folder structure
        emotions = sorted(os.listdir(TRAIN_DIR))
        return X_train, X_test, y_train, y_test, emotions

    print("[INFO] Cache not found. Please run pca_feature_selection.py or wait for extraction...")
    # Fallback extraction if necessary (Copied logic for standalone safety)
    emotions = sorted(os.listdir(TRAIN_DIR))
    label_map = {e: i for i, e in enumerate(emotions)}
    
    def get_imgs(d):
        X, y = [], []
        for e in emotions:
            p = os.path.join(d, e)
            if not os.path.exists(p): continue
            l = label_map[e]
            for f in os.listdir(p):
                fp = os.path.join(p, f)
                img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
                X.append(img)
                y.append(l)
        return np.array(X), np.array(y)

    X_tr_img, y_tr = get_imgs(TRAIN_DIR)
    X_te_img, y_te = get_imgs(TEST_DIR)
    
    def ext_hog(imgs):
        return np.array([hog(i, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys') for i in imgs])
        
    X_tr = ext_hog(X_tr_img)
    X_te = ext_hog(X_te_img)
    
    # Save cache
    np.save(TRAIN_HOG_FILE, X_tr)
    np.save(TEST_HOG_FILE, X_te)
    np.save(TRAIN_LABELS_FILE, y_tr)
    np.save(TEST_LABELS_FILE, y_te)
    
    return X_tr, X_te, y_tr, y_te, emotions

# ==========================================
# MODELS DEFINITION
# ==========================================
# Defining models map
models_map = {
    "SVM": SVC(kernel='linear', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# ==========================================
# HELPER: PLOT CONFUSION MATRIX
# ==========================================
def save_confusion_matrix(cm, classes, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ==========================================
# HELPER: PLOT ROC CURVE
# ==========================================
def save_roc_curve(model, X_test, y_test, n_classes, classes, name):
    # Try to get evaluation probabilities
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        print(f"Warning: {name} does not support probability prediction for ROC.")
        return

    # Binarize output
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        # Handle cases where model output might process differently
        if y_score.shape[1] == n_classes:
             fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
             roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label='ROC class {0} (area = {1:0.2f})'.format(classes[i], roc_auc.get(i, 0)))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(ROC_DIR, f"roc_{name.replace(' ', '_')}.png"))
    plt.close()

# ==========================================
# MAIN LOOP
# ==========================================
def main():
    X_train, X_test, y_train, y_test, emotions = load_data_and_features()
    
    print("\nData Loaded:")
    print(f" Train: {X_train.shape}")
    print(f" Test:  {X_test.shape}")
    print(f" Classes: {emotions}")
    
    results = []
    best_acc = 0
    best_model = None
    best_model_name = ""

    print("\nStarting Model Training & Evaluation...")
    print("="*60)

    for name, model in models_map.items():
        print(f"Training {name}...")
        start_time = time.time()
        
        # Train
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f" -> Accuracy: {acc:.4f} | Time: {train_time:.2f}s")
        
        # Store results
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Time (s)": train_time
        })
        
        # Save Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_path = os.path.join(CM_DIR, f"cm_{name.replace(' ', '_')}.png")
        save_confusion_matrix(cm, emotions, f"Confusion Matrix - {name}", cm_path)
        
        # Save ROC (if possible)
        try:
            save_roc_curve(model, X_test, y_test, len(emotions), emotions, name)
        except Exception as e:
            print(f" [Warn] ROC failed for {name}: {e}")

        # Check Best
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_model_name = name
            
    # ==========================================
    # SAVE RESULTS
    # ==========================================
    print("\n"+"="*60)
    print("FINAL COMPARISON TABLE")
    print("="*60)
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="Accuracy", ascending=False)
    print(df_results.to_string(index=False))
    
    # Save CSV
    df_results.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    
    # Save Best Model
    if best_model:
        model_path = os.path.join(MODELS_DIR, "best_emotion_model.pkl")
        joblib.dump(best_model, model_path)
        print("\n"+"="*60)
        print(f"Best Model: {best_model_name} with {best_acc:.4f} accuracy")
        print(f"Saved to: {model_path}")
        print("="*60)

if __name__ == "__main__":
    main()
