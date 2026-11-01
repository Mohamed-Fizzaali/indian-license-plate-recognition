
from flask import Flask, request, jsonify
from alpr import recognize_for_app
import numpy as np
import cv2

app = Flask(__name__)


@app.route("/recognize", methods=["POST"])
def recognize():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image_bytes = file.read()

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    result = recognize_for_app(image)

    return jsonify({
        "plate_number": result.plate_number,
        "confidence": result.confidence,
        "valid_indian_format": result.valid_indian_format,
        "status": result.processing_status,
        "error": result.error,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
