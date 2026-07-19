import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
# Load dataset
data = np.load("speech_dataset.npz")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

# Create CNN Model
model = Sequential()

# First Convolution Block
model.add(
    Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation='relu',
        input_shape=(40,32,1)
    )
)

model.add(
    MaxPooling2D(
        pool_size=(2,2)
    )
)

# Second Convolution Block
model.add(
    Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation='relu'
    )
)

model.add(
    MaxPooling2D(
        pool_size=(2,2)
    )
)

# -------------------------------
# Classification Part
# -------------------------------

model.add(Flatten())

model.add(
    Dense(
        128,
        activation='relu'
    )
)

model.add(
    Dropout(0.5)
)

# Replace NUM_CLASSES with your number of output classes
model.add(
    Dense(
        8,
        activation='softmax'
    )
)

# Model Summary
model.summary()


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32
)



np.save(
    "history.npy",
    history.history
)