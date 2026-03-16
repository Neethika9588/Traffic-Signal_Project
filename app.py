from flask import Flask, jsonify
import cv2
import numpy as np

app = Flask(__name__)

def calculate_density(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150)

    vehicle_pixels = np.sum(edges > 0)
    density = vehicle_pixels / image.size

    return density


def signal_time(density):

    if density < 0.02:
        return 10
    elif density < 0.05:
        return 20
    elif density < 0.08:
        return 30
    else:
        return 40


@app.route("/")
def analyze():

    results = {}

    for i in range(1,5):

        image = cv2.imread(f"images/lane{i}.jpg")

        if image is None:
            results[f"lane{i}"] = "Image not found"
            continue

        density = calculate_density(image)
        time = signal_time(density)

        results[f"lane{i}"] = {
            "density": float(density),
            "green_signal_time": time
        }

    return jsonify(results)


app.run(host="0.0.0.0", port=10000)
