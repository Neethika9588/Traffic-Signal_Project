from flask import Flask, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------- Vehicle Counting (approximation using contours) --------
def count_vehicles(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    edges = cv2.Canny(blur,50,150)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    vehicle_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 500:   # ignore small noise
            vehicle_count += 1

    return vehicle_count


# -------- Signal Timing Logic --------
def signal_time(vehicle_count):

    if vehicle_count == 0:
        return 0

    elif vehicle_count <= 10:
        return 12      # 10–15 seconds

    elif vehicle_count <= 30:
        return 22      # 20–25 seconds

    elif vehicle_count > 40:
        return 38      # 35–40 seconds

    else:
        return 30


@app.route("/")
def analyze():

    results = {}

    for i in range(1,5):

        image_path = os.path.join(BASE_DIR, f"lane{i}.jpg")

        image = cv2.imread(image_path)

        if image is None:
            results[f"lane{i}"] = "Image not found"
            continue

        vehicles = count_vehicles(image)

        time = signal_time(vehicles)

        results[f"lane{i}"] = {
            "vehicle_count": vehicles,
            "green_signal_time": time
        }

    return jsonify(results)


app.run(host="0.0.0.0", port=10000)
