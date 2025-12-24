import os
import cv2
import matplotlib.pyplot as plt

train_dir = "train"
# Emotion labels (folder names)
emotions = os.listdir(train_dir)
print("Emotions:", emotions)

# Load one image from each class
plt.figure(figsize=(10, 6))

for i, emotion in enumerate(emotions):
    emotion_folder = os.path.join(train_dir, emotion)
    image_name = os.listdir(emotion_folder)[0]  # first image
    image_path = os.path.join(emotion_folder, image_name)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    plt.subplot(2, 4, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(emotion)
    plt.axis('off')

plt.tight_layout()
plt.show()
