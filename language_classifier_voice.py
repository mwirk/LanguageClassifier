from tensorflow.keras.models import load_model
import preprocess_voice
import joblib
import numpy as np
import whisper
import joblib
import tempfile

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