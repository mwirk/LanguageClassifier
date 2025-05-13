from flask import Flask, render_template, request
import language_classifier_voice  

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', result="")

@app.route('/analize_voice', methods=['POST'])
def analize_voice():
    file = request.files['audio']
    result = language_classifier_voice.classify(file)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
