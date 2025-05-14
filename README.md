# 🌍 LanguageClassifier  
Detect spoken language from audio (Spanish 🇪🇸 / English 🇬🇧 / German 🇩🇪 / Russian 🇷🇺)

This project uses [OpenAI Whisper](https://github.com/openai/whisper) to transcribe `.mp3` audio files and a classical machine learning pipeline to classify the **spoken language** based on the **transcription**. It includes a simple **Flask web interface** where users can upload audio files and receive a prediction.

A minimal Flask API allows users to upload .mp3 audio and get a language prediction in real time.



---

## 🛠️ Technologies Used

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)  
![Flask](https://img.shields.io/badge/Flask-2.3-black?logo=flask)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-FF6F00?logo=tensorflow)  
![Whisper](https://img.shields.io/badge/Whisper-Base--Model-4B8BBE?logo=OpenAI&logoColor=white)

---

## 📦 How It Works

### 🗂️ 1. Data Loading

The `load_file_paths_and_labels()` function walks through a directory, collects paths to `.mp3` files.

### 🧠 2. Speech-to-Text with Whisper

```python
whisper_model = whisper.load_model("base")
result = whisper_model.transcribe(path)
```
Loads the Whisper model (base size) from OpenAI, a pre-trained speech-to-text model.
For each audio file, it transcribes speech into text.


## 🧪 3. Preprocessing

Audio is transcribed into text using Whisper.

Transcribed text is then vectorized using TF-IDF for classification.

---

## 🔤 4. Text Classification Pipeline

After transcription and preprocessing:

```python
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)
``` 
Texts (X) and labels (y) are split into training and test sets.

A simple pipeline is defined using TF-IDF and Logistic Regression:
```python
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
    ('clf', LogisticRegression(max_iter=1000))
])
```
TfidfVectorizer: Converts text to numerical vectors using term frequency-inverse document frequency.
LogisticRegression: A simple linear classifier that predicts the language label.

TF (Term Frequency) - How often a word appears.

IDF (Inverse Document Frequency) - How rare a word is across all documents.
Words that are common in one file but rare in others get higher scores, making them more useful for classification.

Logistic Regression uses linear model to calculate the probability that a given input (vectorized text) belongs to a specific class (e.g., English, Spanish). It does this by learning weights for each word feature during training.
It's called “logistic” because it uses the sigmoid function to map values between 0 and 1 (probabilities).


TF-IDF + Logistic Regression is fast, interpretable, and needs less data than deep learning models.

## Pipeline Summary


Audio → Text (via Whisper) → TF-IDF → Logistic Regression

After that, model is ready to classify language by audio. 

Below we can see function which is called in Flask api to detect what language we can hear in the speech.
```python
def classify(file):
    whisper_model = whisper.load_model("base")
    

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        file.save(tmp.name)
        result = whisper_model.transcribe(tmp.name)

    text = result["text"].strip()
    if not text:
        return "Audio was too unclear to transcribe."

    clf = joblib.load("text_lang_classifier.pkl")
    predicted_language = clf.predict([text])[0]

    return f"Predicted language: {predicted_language}\nTranscription: {text}"

```

