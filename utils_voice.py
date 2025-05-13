
import os
import numpy as np
from preprocess_voice import extract_mfcc
from sklearn.preprocessing import LabelEncoder
import joblib  

def load_data(data_dir, encoder_path="label_encoder.pkl"):
    X = []
    y = []
    labels = os.listdir(data_dir)

    for label in labels:
        folder = os.path.join(data_dir, label)
        for file in os.listdir(folder):
            if file.endswith('.mp3'):
                filepath = os.path.join(folder, file)
                mfcc = extract_mfcc(filepath)
                X.append(mfcc)
                y.append(label)

    X = np.array(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    joblib.dump(le, encoder_path)

    return X, y

def preprocess_single_file(file_path, encoder_path="label_encoder.pkl"):
    mfcc = extract_mfcc(file_path)
    mfcc = mfcc[..., np.newaxis]  
    mfcc = np.expand_dims(mfcc, axis=0) 

    le = joblib.load(encoder_path)

    return mfcc, le
