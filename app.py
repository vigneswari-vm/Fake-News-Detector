# app.py
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load saved models
vectorizer = pickle.load(open('tfidf_model.pkl', 'rb'))
model = pickle.load(open('classifier.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']
        data = vectorizer.transform([news])
        prediction = model.predict(data)
        result = "🟢 Real News" if prediction[0] == 'REAL' else "🔴 Fake News"
        return render_template('result.html', prediction=result, news=news)

if __name__ == '__main__':
    app.run(debug=True)
