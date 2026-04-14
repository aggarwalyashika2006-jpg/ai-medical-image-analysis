
# AI‑Powered Medical Image Analysis – Pneumonia Detection

This project is a deep learning–based web application that detects pneumonia from chest X‑ray images using a Convolutional Neural Network (CNN). It provides a simple interface for clinicians or researchers to upload X‑rays and receive an automatic prediction (Pneumonia / Normal). [web:239][web:248][web:250]

---

## Features

- CNN model trained on chest X‑ray images for pneumonia detection. [web:239][web:248][web:250]
- Web interface (Flask) to upload images and view predictions.
- Separate scripts for training, evaluation, and inference.
- Reproducible environment using Python and common deep learning libraries (TensorFlow/Keras, NumPy, etc.). [web:238][web:242][web:247]

---

## Project Structure

```text
Medical_Image_Analysis/
│
├── app.py                # Flask web app for inference
├── train_model.py        # Script to train the CNN model
├── evaluate_model.py     # Script to evaluate the trained model
├── templates/
│   └── index.html        # Frontend template for file upload UI
├── .gitignore
└── README.md
```

(Adjust this tree to match the exact files in the repo.)

---

## Dataset

The project is designed for chest X‑ray–based pneumonia detection, typically using public datasets such as the Kaggle “Chest X‑Ray Images (Pneumonia)” dataset or similar medical imaging datasets. You can download a dataset and place it in a `data/` directory with `train`, `val`, and `test` splits. [web:239][web:244][web:248]

Example structure:

```text
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/aggarwalyashika2006-jpg/ai-medical-image-analysis.git
cd ai-medical-image-analysis
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate  # On Linux/Mac
```

3. Install dependencies (after creating a `requirements.txt`):

```bash
pip install -r requirements.txt
```

---

## Training the Model

1. Ensure your dataset is organized in the `data/` directory as described above.
2. Update dataset paths and hyperparameters in `train_model.py`.
3. Run:

```bash
python train_model.py
```

This trains the CNN and saves the model weights (for example in a `models/` directory). [web:239][web:242][web:248]

---

## Evaluating the Model

After training, evaluate performance on the test set:

```bash
python evaluate_model.py
```

This can report metrics such as accuracy, precision, recall, F1 score, and confusion matrix for pneumonia vs. normal classes. [web:239][web:242]

---

## Running the Web App

1. Make sure the trained model path in `app.py` points to your saved model. [web:238][web:239]
2. Start the Flask server:

```bash
python app.py
```

3. Open your browser and go to:

```text
http://127.0.0.1:5000
```

4. Upload a chest X‑ray image and view the prediction.

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib
- Flask (for web interface)
- HTML/CSS (for frontend template) [web:238][web:239][web:248]

---

## Future Improvements

- Support for additional thoracic diseases and multi‑label classification.
- Improved model explainability (Grad‑CAM or heatmaps to visualize important regions).
- Containerized or cloud deployment for scalable clinical use. [web:239][web:243][web:248]

---

## Disclaimer

This project is for research and educational purposes only and **must not** be used as a standalone diagnostic tool. Clinical decisions should always be made by qualified healthcare professionals. [web:243][web:250]
