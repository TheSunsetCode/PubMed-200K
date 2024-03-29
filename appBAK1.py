from flask import Flask, render_template, request, redirect, url_for
from backend import output_to_text, preprocess_text, process_output, split_into_sentences, DictKeyCombine
from keras.models import load_model
from flask import Markup

class_names= ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS']

model = load_model('Models\\NLP_Sentence_Encoder_Model\\NLP_Sentence_Encoder_Model')
output = None
TextData = None
TEXT = None
result = None

def detect_words(text: str, words: list[str]):
    for word in words:
        if word in text:
            return f"<h2>{word}</h2>"
    return ""

def output_to_html(output):
    if not output:
        return ""
    html = ""
    for sentence in output:
        for key, value in sentence.items():
            html += f"<h2>{key}</h2> <br> <p>{value}</p> <br>"
    return html

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    text = ""
    if request.method == 'POST':
        text = request.form['text']
        main_API(text=text)
    return render_template('index.html', title='PubMed 200K', txt=Markup(output_to_html(TEXT)))

def main_API(text):
    global ERRORMESSAGE
    global output
    global TEXT
    global result
    if request.method == 'POST':
        try:
            data = preprocess_text(text)
            output = process_output(
                data = data,
                sentences = split_into_sentences(text),
                class_names = class_names,
                model = model
            )
            result = DictKeyCombine(output)
            clases, TEXT = output_to_text(output, class_names)
            return ""
        except Exception as e:
            ERRORMESSAGE = str(e)
            return ERRORMESSAGE

if __name__ == '__main__':
    app.run(debug=True)
