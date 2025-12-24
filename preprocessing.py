import os
import cv2
import numpy as np

TRAIN_DIR = "train"
TEST_DIR = "test"

# Fixed image size
IMG_SIZE = 48
emotions = sorted(os.listdir(TRAIN_DIR))

label_map = {}
for idx, emotion in enumerate(emotions):
    label_map[emotion] = idx

print("Label Mapping:")
print(label_map)

def load_data(data_dir):
    X = []
    y = []

    for emotion in emotions:
        folder_path = os.path.join(data_dir, emotion)
        label = label_map[emotion]

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            # Read image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Resize image
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Normalize pixel values (0 → 1)
            img = img / 255.0

            X.append(img)
            y.append(label)

    return np.array(X), np.array(y)


X_train, y_train = load_data(TRAIN_DIR)
X_test, y_test = load_data(TEST_DIR)

print("\nTraining data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)

print("\nTesting data shape:", X_test.shape)
print("Testing labels shape:", y_test.shape)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print("\nFlattened training data shape:", X_train_flat.shape)
print("Flattened testing data shape:", X_test_flat.shape)

print("\nDay 2 preprocessing completed successfully.")
