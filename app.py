import firebase_admin
from firebase_admin import credentials, db
cred = credentials.Certificate("https://smart-traffic-system-4ffbf-default-rtdb.firebaseio.com/")

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://smart-traffic-system-4ffbf-default-rtdb.firebaseio.com'
})
from flask import Flask, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# -------- Vehicle Detection using Image Processing --------
def detect_vehicles(img_path):

    # Check file exists
    if not os.path.exists(img_path):
        return 0

    # Read image
    image = cv2.imread(img_path)

    if image is None:
        return 0

    # Step 1: Resize image
    image = cv2.resize(image, (800,600))

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Remove noise
    blur = cv2.GaussianBlur(gray,(5,5),0)

    # Step 4: Edge detection
    edges = cv2.Canny(blur,50,150)

    # Step 5: Morphological closing to connect vehicle edges
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Step 6: Find contours
    contours, _ = cv2.findContours(
        closing,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    vehicle_count = 0

    # Step 7: Filter contours by size
    for cnt in contours:

        area = cv2.contourArea(cnt)

        if 500 < area < 50000:
            vehicle_count += 1

    return vehicle_count


# -------- Traffic Signal Timing Rule --------
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


# -------- Analyze All 4 Lanes --------
@app.route("/")
def analyze():

    results = {}

    for i in range(1,5):

        img_path = os.path.join(BASE_DIR, f"lane{i}.jpg")

        vehicles = detect_vehicles(img_path)

        time = signal_time(vehicles)

        results[f"lane{i}"] = {
            "vehicle_count": vehicles,
            "green_signal_time": time
        }
    ref = db.reference("traffic_data")
    ref.set(results)

    return jsonify(results)


# -------- Run Flask Server --------
app.run(host="0.0.0.0", port=10000)
