from flask import Flask, render_template, request
from backend import preprocess_text
import keras
from keras.models import load_model

# AI
# define your custom 'TextVectorization' class here

# load the saved model within a CustomObjectScope that includes your custom class

model = load_model('Models\\NLP_Sentence_Encoder_Model\\NLP_Sentence_Encoder_Model')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        # do something with the text here
        data = preprocess_text(text)
        output = model.predict(data)
    else:
        text = ''
    return render_template('index.html', title='PubMed 200K', text=text)

if __name__ == '__main__':
    app.run(debug=True)
