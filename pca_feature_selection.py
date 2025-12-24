import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# ==========================================
# CONFIGURATION
# ==========================================
TRAIN_DIR = "train"
TEST_DIR = "test"
IMG_SIZE = 48
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Cache files
TRAIN_HOG_FILE = "X_train_hog.npy"
TEST_HOG_FILE = "X_test_hog.npy"
TRAIN_LABELS_FILE = "y_train.npy"
TEST_LABELS_FILE = "y_test.npy"

# ==========================================
# DATA LOADING & HOG EXTRACTION
# ==========================================
def load_and_extract():
    # Check if cached
    if all(os.path.exists(f) for f in [TRAIN_HOG_FILE, TEST_HOG_FILE, TRAIN_LABELS_FILE, TEST_LABELS_FILE]):
        print("[INFO] Loading cached features...")
        X_train_hog = np.load(TRAIN_HOG_FILE)
        X_test_hog = np.load(TEST_HOG_FILE)
        y_train = np.load(TRAIN_LABELS_FILE)
        y_test = np.load(TEST_LABELS_FILE)
        return X_train_hog, X_test_hog, y_train, y_test

    # If not cached, we need label map to be consistent. 
    # We'll assume the same sorting as classification script.
    emotions = sorted(os.listdir(TRAIN_DIR))
    label_map = {emotion: idx for idx, emotion in enumerate(emotions)}
    print(f"[INFO] Label Map: {label_map}")

    def get_data(data_dir):
        X_data = []
        y_data = []
        for emotion in emotions:
            folder_path = os.path.join(data_dir, emotion)
            label = label_map[emotion]
            # Verify folder exists
            if not os.path.exists(folder_path): continue
            
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X_data.append(img)
                y_data.append(label)
        return np.array(X_data), np.array(y_data)

    print("[INFO] Loading images and extracting features. This may take a while...")
    X_train_imgs, y_train = get_data(TRAIN_DIR)
    X_test_imgs, y_test = get_data(TEST_DIR)

    def extract_hog(images):
        feats = []
        for img in images:
            # Normalizing image before HOG
            img = img / 255.0
            fd = hog(img, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2), block_norm='L2-Hys')
            feats.append(fd)
        return np.array(feats)

    X_train_hog = extract_hog(X_train_imgs)
    X_test_hog = extract_hog(X_test_imgs)

    # Save cache
    np.save(TRAIN_HOG_FILE, X_train_hog)
    np.save(TEST_HOG_FILE, X_test_hog)
    np.save(TRAIN_LABELS_FILE, y_train)
    np.save(TEST_LABELS_FILE, y_test)
    
    return X_train_hog, X_test_hog, y_train, y_test

# ==========================================
# MAIN EXECUTION
# ==========================================

print("========================================")
print("   PCA FEATURE SELECTION ANALYSIS")
print("========================================")

X_train, X_test, y_train, y_test = load_and_extract()
print(f"Original Feature Shape: {X_train.shape}")

# 1. Baseline (No PCA)
print("\n[TEST] Baseline Linear SVM (Assessment)...")
# Using a faster/simpler SVM for speed in comparison or Full Linear SVC
baseline_model = SVC(kernel='linear', random_state=42)
baseline_model.fit(X_train, y_train)
base_preds = baseline_model.predict(X_test)
base_acc = accuracy_score(y_test, base_preds)
print(f" -> Baseline Accuracy: {base_acc:.4f}")

# 2. PCA Analysis
components_list = [50, 100, 200, 300]
results = {}
best_acc = 0
best_pca_model = None
best_n = 0

print("\n[TEST] Testing PCA Components...")

for n in components_list:
    print(f"\n--- n_components = {n} ---")
    
    # Fit PCA
    pca = PCA(n_components=n, whiten=True, random_state=42)
    pca.fit(X_train)
    
    # Transform
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f" Explained Variance Ratio: {explained_var:.4f}")
    
    # Train SVM on reduced features
    clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42) # RBF usually better with PCA
    clf.fit(X_train_pca, y_train)
    preds = clf.predict(X_test_pca)
    acc = accuracy_score(y_test, preds)
    
    print(f" Model Accuracy: {acc:.4f}")
    
    results[n] = {'variance': explained_var, 'accuracy': acc}
    
    # Track best
    if acc > best_acc:
        best_acc = acc
        best_pca_model = pca
        best_n = n

print("\n========================================")
print(" SUMMARY RESULTS")
print("========================================")
print(f"{'Components':<15} | {'Explained Var':<15} | {'Accuracy':<15}")
print("-" * 50)
print(f"{'Full (HOG)':<15} | {'1.0000':<15} | {base_acc:.4f}")
for n in components_list:
    res = results[n]
    print(f"{n:<15} | {res['variance']:.4f}          | {res['accuracy']:.4f}")

print("-" * 50)
print(f"Best PCA Performance: {best_acc:.4f} with {best_n} components")

# Save Best PCA
pca_path = os.path.join(MODELS_DIR, "best_pca.pkl")
if best_pca_model:
    joblib.dump(best_pca_model, pca_path)
    print(f"\n[INFO] Best PCA model saved to {pca_path}")
else:
    print("\n[INFO] PCA did not improve over baseline (or failed).")

print("Done.")
