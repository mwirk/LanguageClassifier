
import librosa
import numpy as np

def extract_mfcc(file_path, n_mfcc=40, max_pad_len=862):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    return mfcc


def preprocess_audio(file_path, max_pad_len=862):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]  

    mfcc = mfcc[..., np.newaxis]     
    mfcc = np.expand_dims(mfcc, 0)  
    return mfcc


