# 🌍 LanguageClassifier  
Detect spoken language from audio (Spanish 🇪🇸 / English 🇬🇧 / German 🇩🇪 / Russian 🇷🇺)

This project uses [OpenAI Whisper](https://github.com/openai/whisper) to transcribe `.mp3` audio files and a classical machine learning pipeline to classify the **spoken language** based on the **transcription**. It includes a simple **Flask web interface** where users can upload audio files and receive a prediction.

Web Interface (Flask)

A minimal Flask API allows users to:

Upload .mp3 audio
Get a language prediction in real time



> 💡 A lightweight and interpretable solution for speech-based language classification.

---

## 🛠️ Technologies Used

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)  
![Flask](https://img.shields.io/badge/Flask-2.3-black?logo=flask)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-FF6F00?logo=tensorflow)  
![Whisper](https://img.shields.io/badge/Whisper-Base--Model-4B8BBE?logo=OpenAI&logoColor=white)

---

## 📦 How It Works

### 🗂️ 1. Data Loading

The `load_file_paths_and_labels()` function walks through a directory, collects paths to `.mp3` files, and infers **labels** from folder names:

audio_data/
├── english/
│ ├── file1.mp3
├── spanish/
│ ├── file2.mp3

### 🧠 2. Speech-to-Text with Whisper

```python
whisper_model = whisper.load_model("base")
result = whisper_model.transcribe(path)
```
## 🧪 3. Preprocessing

- **MFCC features** are extracted from audio files during preprocessing. These features capture the short-term power spectrum of sound and are widely used in speech and audio recognition tasks.
- Transcribed text from the Whisper model is also collected for each audio file to be used in the classification step.

---

## 🔤 4. Text Classification Pipeline

After transcription and preprocessing:

```python
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)
``` 


A simple pipeline is defined using TF-IDF and Logistic Regression:
```python
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
    ('clf', LogisticRegression(max_iter=1000))
])
```
TfidfVectorizer: Converts text to numerical vectors using term frequency-inverse document frequency.
LogisticRegression: A simple linear classifier that predicts the language label.

## Pipeline Summary

Audio (.mp3)
   ↓
Whisper (Speech-to-Text)
   ↓
Text
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression (Classification)
   ↓
Language Prediction (Spanish/English/German/Russian)
