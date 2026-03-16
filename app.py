from flask import Flask, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- Vehicle Detection ----------
def detect_density(img_path):

    if not os.path.exists(img_path):
        return 0

    img = cv2.imread(img_path)

    if img is None:
        return 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (7,7), 0)

    edges = cv2.Canny(blur, 50, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vehicle_count = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > 1500:
            vehicle_count += 1

    return vehicle_count


# ---------- Signal Timing Rule ----------
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


# ---------- Analyze Lanes ----------
@app.route("/")
def analyze():

    results = {}

    for i in range(1,5):

        image_path = os.path.join(BASE_DIR, f"lane{i}.jpg")

        vehicles = detect_density(image_path)

        time = signal_time(vehicles)

        results[f"lane{i}"] = {
            "vehicle_count": vehicles,
            "green_signal_time": time
        }

    return jsonify(results)


app.run(host="0.0.0.0", port=10000)
