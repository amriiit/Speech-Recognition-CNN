import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATASET_PATH = "Dataset"

X = []
y = []

for label in os.listdir(DATASET_PATH):

    folder_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(folder_path):
        continue

    print(f"Processing {label}")

    for file in os.listdir(folder_path):

        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(folder_path, file)

        audio, sr = librosa.load(file_path, sr=16000)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        )
        # Target number of time frames
        MAX_LEN = 32

        # If MFCC has fewer than 32 columns, pad with zeros
        if mfcc.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(
                mfcc,
                pad_width=((0, 0), (0, pad_width)),
                mode='constant'
            )

        # If MFCC has more than 32 columns, cut the extra columns
        else:
            mfcc = mfcc[:, :MAX_LEN]

        X.append(mfcc)
        y.append(label)
X = np.array(X)
y = np.array(y)
X = X[..., np.newaxis]
print("X shape:", X.shape)
encoder = LabelEncoder()
y = encoder.fit_transform(y)
np.save("classes.npy", encoder.classes_)
print("\nLabel Mapping:")
for i, label in enumerate(encoder.classes_):
    print(f"{label} -> {i}")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
np.savez(
    "speech_dataset.npz",
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)
print("\nTraining Shape:", X_train.shape)
print("Testing Shape :", X_test.shape)