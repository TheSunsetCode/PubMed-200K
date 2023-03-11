from flask import Flask, render_template, request, redirect, url_for
from backend import output_to_text, preprocess_text, process_output, split_into_sentences
import keras
from keras.models import load_model
import requests

class_names= ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']

model = load_model('Models\\NLP_Sentence_Encoder_Model\\NLP_Sentence_Encoder_Model')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        return redirect(url_for('main_API', text=text))
    return render_template('index.html', title='PubMed 200K')

@app.route('/api/v1/main', methods=['GET', 'POST'])
def main_API():
    global ERRORMESSAGE
    global output
    global text
    if request.method == 'POST':
        text = request.form['text']
        try:
            data = preprocess_text(text)
            output = process_output(
                data = data,
                sentences = split_into_sentences(text),
                class_names = class_names,
                model = model
            )
            clases, text = output_to_text(output, class_names)
            return redirect(url_for('home'))
        except Exception as e:
            ERRORMESSAGE = str(e)
            return ERRORMESSAGE
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)

