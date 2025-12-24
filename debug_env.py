import sys
import os
print(sys.executable)
print(sys.path)
try:
    import cv2
    print("cv2 imported successfully")
except ImportError as e:
    print(f"ImportError: {e}")
