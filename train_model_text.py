import os
import whisper
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_file_paths_and_labels(audio_dir):
    file_paths = []
    labels = []
    for root, dirs, files in os.walk(audio_dir):
        for f in files:
            if f.endswith(".mp3"):
                label = os.path.basename(root) 
                file_paths.append(os.path.join(root, f))
                labels.append(label)
    return file_paths, labels

audio_dir = "audio_data/"  
file_paths, labels = load_file_paths_and_labels(audio_dir)



whisper_model = whisper.load_model("base") 

texts = []
final_labels = []


for path, label in zip(file_paths, labels):
    try:
        result = whisper_model.transcribe(path)
        transcription = result["text"].strip()
        if transcription:
            texts.append(transcription)
            final_labels.append(label)
        else:
            print(f"No transcription from {path}")
    except Exception as e:
        print(f"Failed to transcribe {path}: {e}")



X_train, X_test, y_train, y_test = train_test_split(texts, final_labels, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)


print(classification_report(y_test, y_pred))


joblib.dump(pipeline, 'text_lang_classifier.pkl')

