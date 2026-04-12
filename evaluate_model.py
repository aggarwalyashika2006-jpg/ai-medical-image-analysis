import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

IMG_SIZE = 256
BATCH_SIZE = 32

test_dir = "chest_xray/test"

datagen = ImageDataGenerator(rescale=1.0 / 255)

test_data = datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

model = tf.keras.models.load_model("medical_ai_model.h5")

y_true = test_data.classes
y_pred_prob = model.predict(test_data)
y_pred = np.round(y_pred_prob).astype(int).ravel()

acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print(f"Model Accuracy: {acc * 100:.2f}%")
print("Confusion Matrix:")
print(cm)