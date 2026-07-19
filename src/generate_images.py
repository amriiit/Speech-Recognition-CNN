import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

os.makedirs("images", exist_ok=True)

# ----------------------------
# Load model
# ----------------------------
model = tf.keras.models.load_model("models/speech_cnn.keras")

# ----------------------------
# Load dataset
# ----------------------------
data = np.load("speech_dataset.npz")

X_test = data["X_test"]
y_test = data["y_test"]

classes = np.load("classes.npy")

# ----------------------------
# Architecture
# ----------------------------
tf.keras.utils.plot_model(
    model,
    to_file="images/architecture.png",
    show_shapes=True,
    show_layer_names=True,
    dpi=300
)

print("✓ Architecture saved")

# ----------------------------
# Confusion Matrix
# ----------------------------
predictions = model.predict(X_test)

predicted_labels = np.argmax(predictions, axis=1)

cm = confusion_matrix(y_test, predicted_labels)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

plt.figure(figsize=(8,8))

disp.plot()

plt.title("Confusion Matrix")

plt.savefig(
    "images/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("✓ Confusion Matrix saved")

# ----------------------------
# Accuracy & Loss
# ----------------------------

history = np.load(
    "history.npy",
    allow_pickle=True
).item()

# Accuracy

plt.figure(figsize=(8,5))

plt.plot(history["accuracy"], linewidth=2)

plt.plot(history["val_accuracy"], linewidth=2)

plt.legend(["Training","Validation"])

plt.title("Training Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.grid(True)

plt.savefig(
    "images/accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("✓ Accuracy graph saved")

# Loss

plt.figure(figsize=(8,5))

plt.plot(history["loss"], linewidth=2)

plt.plot(history["val_loss"], linewidth=2)

plt.legend(["Training","Validation"])

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid(True)

plt.savefig(
    "images/loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("✓ Loss graph saved")