import numpy as np
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
wav = np.load("./dataset_catch/train/clean/p226_001.wav_1.npy")

print(wav.shape)
spec = np.load("./dataset_spec_catch/train/noisy/p226_001.wav_1.npy")
melspec = librosa.feature.melspectrogram(y=wav, sr=16000, n_fft=512, hop_length=1024, win_length=25, n_mels=128)
logmelspec = librosa.power_to_db(melspec)
print(logmelspec.shape)

plt.figure()
librosa.display.specshow(spec, sr=16000, x_axis='time', y_axis='mel')
plt.show()