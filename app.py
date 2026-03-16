from flask import Flask, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# -------- Image Processing Pipeline --------
def process_image(img_path):

    if not os.path.exists(img_path):
        return 0

    img = cv2.imread(img_path)

    if img is None:
        return 0

    # Step 1: Resize image
    img = cv2.resize(img, (800, 600))

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Noise removal
    blur = cv2.GaussianBlur(gray, (7,7), 0)

    # Step 4: Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    # Step 5: Morphological operations
    kernel = np.ones((5,5), np.uint8)
    dilation = cv2.dilate(thresh, kernel, iterations=2)

    # Step 6: Find contours
    contours, _ = cv2.findContours(
        dilation,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    vehicle_count = 0

    # Step 7: Filter contours by size
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 2000:
            vehicle_count += 1

    return vehicle_count


# -------- Signal Timing Rule --------
def signal_time(vehicle_count):

    if vehicle_count == 0:
        return 0

    elif vehicle_count <= 10:
        return 12

    elif vehicle_count <= 30:
        return 22

    elif vehicle_count > 40:
        return 38

    else:
        return 30


# -------- Analyze Lanes --------
@app.route("/")
def analyze():

    results = {}

    for i in range(1,5):

        img_path = os.path.join(BASE_DIR, f"lane{i}.jpg")

        vehicles = process_image(img_path)

        time = signal_time(vehicles)

        results[f"lane{i}"] = {
            "vehicle_count": vehicles,
            "green_signal_time": time
        }

    return jsonify(results)


app.run(host="0.0.0.0", port=10000)
