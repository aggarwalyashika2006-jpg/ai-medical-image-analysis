from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2

IMG_SIZE = 256

app = Flask(__name__)

model = tf.keras.models.load_model("medical_ai_model.h5")

def preprocess_image(file_bytes):
    image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
    image = image.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return image
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    img_array = preprocess_image(file.read())

    pred = model.predict(img_array)[0][0]
    label = "Pneumonia Detected" if pred > 0.5 else "Normal"
    confidence = float(pred if pred > 0.5 else 1 - pred)

    return jsonify({
        "prediction": label,
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)