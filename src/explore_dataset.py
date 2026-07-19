import librosa
import librosa.display
import matplotlib.pyplot as plt

audio_path = "Dataset/wow/0a0b46ae_nohash_0.wav"

audio, sr = librosa.load(audio_path, sr=None)

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=13
)

print("MFCC Shape:", mfcc.shape)

plt.figure(figsize=(12,6))

librosa.display.specshow(
    mfcc,
    x_axis='time'
)

plt.colorbar()

plt.title("MFCC")

plt.tight_layout()

plt.show()