import numpy as np
import librosa
import tensorflow as tf

model = tf.keras.models.load_model("models/speech_cnn.keras")
classes = np.load("classes.npy")
audio_path = input("Enter audio file path: ")
audio, sr = librosa.load(
    audio_path,
    sr=16000
)
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=40
)

MAX_LEN = 32
if mfcc.shape[1] < MAX_LEN:

    pad_width = MAX_LEN - mfcc.shape[1]

    mfcc = np.pad(
        mfcc,
        pad_width=((0, 0), (0, pad_width)),
        mode="constant"
    )
else:

   mfcc = mfcc[:, :MAX_LEN]

mfcc = mfcc[..., np.newaxis]
mfcc = np.expand_dims(mfcc, axis=0)

prediction = model.predict(mfcc)

predicted_index = np.argmax(prediction)

predicted_word = classes[predicted_index]

confidence = prediction[0][predicted_index] * 100

print("\nPrediction Probabilities:")

for i in range(len(classes)):
    print(f"{classes[i]} : {prediction[0][i]*100:.2f}%")

print("\n--------------------------------")

print(f"Predicted Word : {predicted_word}")

print(f"Confidence     : {confidence:.2f}%")