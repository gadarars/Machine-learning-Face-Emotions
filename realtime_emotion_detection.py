import cv2
import numpy as np
import joblib
import os
from skimage.feature import hog

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "models/best_emotion_model.pkl"
TRAIN_DIR = "train" # To read class names
IMG_SIZE = 48

# Load Haar Cascade
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

def extract_features(image):
    # Same parameters as training
    fd = hog(image, orientations=9, pixels_per_cell=(8, 8),
             cells_per_block=(2, 2), block_norm='L2-Hys')
    return fd.reshape(1, -1)


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        return

    print(f"[INFO] Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    if os.path.exists(TRAIN_DIR):
        emotions = sorted(os.listdir(TRAIN_DIR))
    else:
        emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    print(f"[INFO] Classes: {emotions}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Starting Webcam. Press 'q' to quit.")

    frame_count = 0
    last_prediction = "neutral"
    last_confidence = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for speed
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Faster detection parameters
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(48, 48))

        for (x, y, w, h) in faces:
            # Process every 5th frame only (makes it 5x faster!)
            if frame_count % 5 == 0:
                roi_gray = gray[y:y + h, x:x + w]
                roi_resized = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE))
                roi_norm = roi_resized / 255.0
                features = extract_features(roi_norm)

                try:
                    prediction = model.predict(features)[0]
                    last_prediction = emotions[prediction]

                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(features)[0]
                        last_confidence = np.max(proba) * 100
                except:
                    pass

            # Use last prediction for smooth display
            text = f"{last_prediction}: {last_confidence:.1f}%"
            color = (0, 255, 0) if last_prediction in ["happy", "surprise", "neutral"] else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        frame_count += 1
        cv2.imshow('Realtime Emotion Detection - Press Q to Quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam closed.")
if __name__ == "__main__":
    main()
