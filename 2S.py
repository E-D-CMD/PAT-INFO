from flask import Flask, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

# Folder to store received faces
SAVE_DIR = "server-faces"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.route("/upload-face", methods=["POST"])
def upload_face():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    # Convert image bytes to OpenCV format
    image_bytes = file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Save image
    filename = f"face_{len(os.listdir(SAVE_DIR))}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(filepath, img)

    return jsonify({"message": "Face received", "file": filename})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
