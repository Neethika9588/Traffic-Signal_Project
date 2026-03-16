from flask import Flask, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def detect_vehicles(img_path):

    if not os.path.exists(img_path):
        return 0

    image = cv2.imread(img_path)

    if image is None:
        return 0

    # Resize for consistent processing
    image = cv2.resize(image, (800,600))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    edges = cv2.Canny(blur,50,150)

    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        closing,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    vehicle_count = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if 500 < area < 50000:
            vehicle_count += 1

    return vehicle_count


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

    return jsonify(results)


app.run(host="0.0.0.0", port=10000)
