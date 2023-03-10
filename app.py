from flask import Flask, render_template, request
from backend import preprocess_text

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        # do something with the text here
        data = preprocess_text(text)
    else:
        text = ''
    return render_template('index.html', title='PubMed 200K', text=text)

if __name__ == '__main__':
    app.run(debug=True)
